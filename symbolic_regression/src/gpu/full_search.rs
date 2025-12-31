#![allow(clippy::too_many_arguments)]

use std::mem;

use bytemuck::{Pod, Zeroable};
use dynamic_expressions::OpId;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::traits::{LookupError, OperatorSet};

use super::{
    END_TOKEN, GpuInitError, KIND_CONST, KIND_END, KIND_OP, KIND_VAR, MAX_CONSTS, MAX_NODES, PackedProgram,
    map_op_to_gpu,
};
use crate::dataset::{Dataset, TaggedDataset};
use crate::hall_of_fame::HallOfFame;
use crate::loss_functions::baseline_loss_from_zero_expression;
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember};
use crate::search_utils::SearchResult;

/// Config for the "full GPU" search path.
///
/// This is intentionally GPU-centric and does **not** try to replicate every
/// detail of the CPU regularized evolution implementation. The priority is:
///
/// - Keep the entire population (programs + constants + losses) on the GPU.
/// - Avoid per-evaluation CPU<->GPU round-trips.
/// - Run a simple, massively-batched evolutionary loop.
///
/// The CPU only dispatches compute kernels and reads back the best program at the end.
#[derive(Copy, Clone, Debug)]
pub struct GpuFullSearchConfig {
    /// How many generations to run.
    pub generations: usize,
    /// Tournament size for parent selection.
    pub tournament_size: u32,
    /// Point-mutation probability for each child (0..1).
    pub mutation_rate: f32,
    /// Probability to ignore the parent and generate a random program (0..1).
    pub regen_rate: f32,

    /// Per-child Adam iterations for constant optimization.
    ///
    /// Set to 0 to disable constant optimization (fastest).
    pub adam_iters: u32,
    pub adam_lr: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub adam_step_clip: f32,

    /// Stddev for mutating constants (only used when mutation happens).
    pub const_mut_sigma: f32,

    /// RNG seed.
    pub seed: u32,
}

impl Default for GpuFullSearchConfig {
    fn default() -> Self {
        Self {
            generations: 2_000,
            tournament_size: 8,
            mutation_rate: 0.35,
            regen_rate: 0.05,
            adam_iters: 0,
            adam_lr: 0.05,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            adam_step_clip: 10.0,
            const_mut_sigma: 0.25,
            seed: 0x1234_5678,
        }
    }
}

/// A GPU-resident evolutionary loop that keeps populations and losses entirely on-device.
///
/// Usage:
/// - allocate `population_total = options.populations * options.population_size`
/// - `init_and_eval` generates random programs + constants and computes initial losses
/// - repeated `evolve_generation` steps update the population
/// - CPU reads back (best_loss_bits, best_index) once at the end, then copies the best program+consts for decoding.
struct GpuFullSearcher {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline_init: wgpu::ComputePipeline,
    pipeline_evolve: wgpu::ComputePipeline,

    bind_group: wgpu::BindGroup,

    // population buffers (two halves: parity 0 then parity 1)
    programs_buf: wgpu::Buffer,
    consts_buf: wgpu::Buffer,
    loss_buf: wgpu::Buffer,

    // state (atomics)
    state_buf: wgpu::Buffer,

    // uniforms
    params_buf: wgpu::Buffer,

    // readbacks
    state_readback: wgpu::Buffer,
    best_readback: wgpu::Buffer,

    // sizes
    n_rows: u32,
    n_features: u32,
    pop_total: u32,
    pop_per_island: u32,
    sum_w: f32,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SearchParams {
    // u0 = (n_rows, n_features, pop_total, pop_per_island)
    u0: [u32; 4],
    // u1 = (gen, parity, tournament_size, adam_iters)
    u1: [u32; 4],
    // u2 = (allowed_unary_mask, allowed_binary_mask, mutate_rate_ppm, regen_rate_ppm)
    u2: [u32; 4],
    // u3 = (seed, 0, 0, 0)
    u3: [u32; 4],
    // f0 = (sum_w, lr, beta1, beta2)
    f0: [f32; 4],
    // f1 = (eps, step_clip, const_sigma, 0)
    f1: [f32; 4],
}

impl GpuFullSearcher {
    fn new(dataset: &Dataset<f32>, pop_total: usize, pop_per_island: usize) -> Result<Self, GpuInitError> {
        let n_rows = dataset.n_rows as u32;
        let n_features = dataset.n_features as u32;

        // Build weights buffer and sum_w.
        let mut w_host: Vec<f32> = vec![1.0; dataset.n_rows];
        if let Some(w) = dataset.weights_slice() {
            w_host.copy_from_slice(w);
        }
        let sum_w: f32 = w_host.iter().sum();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        })
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .map_err(|_| GpuInitError::NoAdapter)?;

        let downlevel = adapter.get_downlevel_capabilities();
        if !downlevel.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            return Err(GpuInitError::NoCompute);
        }

        // Metal often needs this explicitly to allow 7 storage buffers in compute.
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_storage_buffers_per_shader_stage = 8;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu-full-search-device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|_| GpuInitError::RequestDeviceFailed)?;

        // Dataset buffers.
        let x_bytes = bytemuck::cast_slice(dataset.x.as_slice().unwrap());
        let y_bytes = bytemuck::cast_slice(dataset.y_slice());
        let w_bytes = bytemuck::cast_slice(&w_host);

        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x"),
            size: x_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&x_buf, 0, x_bytes);

        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: y_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&y_buf, 0, y_bytes);

        let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w"),
            size: w_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&w_buf, 0, w_bytes);

        // Population buffers. We store two generations back-to-back so we can double-buffer
        // without extra bindings.
        let pop_total_u32 = pop_total as u32;
        let prog_bytes = (2 * pop_total * MAX_NODES) as u64 * mem::size_of::<u32>() as u64;
        let const_bytes = (2 * pop_total * MAX_CONSTS) as u64 * mem::size_of::<f32>() as u64;
        let loss_bytes = (2 * pop_total) as u64 * mem::size_of::<f32>() as u64;

        let programs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("programs"),
            size: prog_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let consts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("consts"),
            size: const_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loss"),
            size: loss_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // State: best_loss_bits + best_index (two u32 atomics). Use 16 bytes for safe alignment.
        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params uniform.
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: mem::size_of::<SearchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Readbacks.
        let state_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let best_readback_size =
            (MAX_NODES * mem::size_of::<u32>()) + (MAX_CONSTS * mem::size_of::<f32>()) + mem::size_of::<f32>();
        let best_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("best_readback"),
            size: best_readback_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shaders + pipelines.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("search_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("search.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("search_bgl"),
            entries: &[
                // x
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // y
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // w
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // programs
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // consts
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // loss
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // state
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("search_pl"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline_init = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("init_and_eval"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("init_and_eval"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_evolve = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evolve_generation"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evolve_generation"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("search_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: programs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: consts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: loss_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline_init,
            pipeline_evolve,
            bind_group,
            programs_buf,
            consts_buf,
            loss_buf,
            state_buf,
            params_buf,
            state_readback,
            best_readback,
            n_rows,
            n_features,
            pop_total: pop_total_u32,
            pop_per_island: pop_per_island as u32,
            sum_w,
        })
    }

    fn write_state_reset(&self) {
        // best_loss_bits = max finite f32 (0x7f7fffff)
        // best_index = 0
        let init: [u32; 4] = [0x7f7f_ffff, 0, 0, 0];
        self.queue.write_buffer(&self.state_buf, 0, bytemuck::cast_slice(&init));
    }

    fn write_params(&self, params: &SearchParams) {
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, workgroups_x: u32) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("search_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("search_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    fn read_state_best(&self) -> (u32, u32) {
        // Copy state -> readback
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("state_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.state_buf, 0, &self.state_readback, 0, 16);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.state_readback.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        pollster::block_on(rx.receive())
            .expect("map callback runs")
            .expect("map ok");

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        let best_loss_bits = words[0];
        let best_index = words[1];
        drop(data);
        self.state_readback.unmap();
        (best_loss_bits, best_index)
    }

    fn read_best_program(&self, best_index: u32) -> PackedProgram {
        let prog_bytes = (MAX_NODES * mem::size_of::<u32>()) as u64;
        let const_bytes = (MAX_CONSTS * mem::size_of::<f32>()) as u64;
        let loss_bytes = mem::size_of::<f32>() as u64;

        let prog_off = best_index as u64 * prog_bytes;
        let const_off = best_index as u64 * const_bytes;
        let loss_off = best_index as u64 * loss_bytes;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("best_copy"),
        });

        // programs -> best_readback[0..prog_bytes]
        encoder.copy_buffer_to_buffer(&self.programs_buf, prog_off, &self.best_readback, 0, prog_bytes);
        // consts -> best_readback[prog_bytes..prog_bytes+const_bytes]
        encoder.copy_buffer_to_buffer(
            &self.consts_buf,
            const_off,
            &self.best_readback,
            prog_bytes,
            const_bytes,
        );
        // loss -> best_readback[prog_bytes+const_bytes..]
        encoder.copy_buffer_to_buffer(
            &self.loss_buf,
            loss_off,
            &self.best_readback,
            prog_bytes + const_bytes,
            loss_bytes,
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = self.best_readback.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        pollster::block_on(rx.receive())
            .expect("map callback runs")
            .expect("map ok");

        let data = slice.get_mapped_range();
        let prog_words: &[u32] = bytemuck::cast_slice(&data[0..prog_bytes as usize]);
        let const_words: &[f32] = bytemuck::cast_slice(&data[prog_bytes as usize..(prog_bytes + const_bytes) as usize]);
        let mut program = [END_TOKEN; MAX_NODES];
        for (i, v) in prog_words.iter().take(MAX_NODES).enumerate() {
            program[i] = *v;
        }

        let mut consts = [0.0f32; MAX_CONSTS];
        for (i, v) in const_words.iter().take(MAX_CONSTS).enumerate() {
            consts[i] = *v;
        }

        drop(data);
        self.best_readback.unmap();

        PackedProgram { program, consts }
        // We return just program+consts; caller can recompute loss if desired.
        // (loss is also read back; keeping API compatible for now)
    }
}

fn masks_from_options<Ops, const D: usize>(options: &Options<f32, D>) -> (u32, u32)
where
    Ops: OperatorSet<T = f32>,
{
    let mut unary: u32 = 0;
    let mut binary: u32 = 0;

    for ops in options.operators.ops_by_arity.iter() {
        for &op in ops {
            if let Some((a, code)) = map_op_to_gpu::<Ops>(op) {
                if a == 1 {
                    unary |= 1u32 << (code as u32);
                } else if a == 2 {
                    binary |= 1u32 << (code as u32);
                }
            }
        }
    }

    // If user disabled everything in a category, keep at least something to avoid infinite loops.
    if unary == 0 {
        // NEG
        unary = 1;
    }
    if binary == 0 {
        // ADD
        binary = 1;
    }

    (unary, binary)
}

fn gpu_opcode_to_name(arity: u8, opcode: u16) -> Option<&'static str> {
    match (arity, opcode) {
        // unary
        (1, 0) => Some("neg"),
        (1, 1) => Some("sin"),
        (1, 2) => Some("cos"),
        (1, 3) => Some("exp"),
        (1, 4) => Some("log"),
        (1, 5) => Some("sqrt"),
        // binary
        (2, 0) => Some("add"),
        (2, 1) => Some("sub"),
        (2, 2) => Some("mul"),
        (2, 3) => Some("div"),
        _ => None,
    }
}

fn unpack_program_to_expr<Ops, const D: usize>(packed: &PackedProgram) -> Result<PostfixExpr<f32, Ops, D>, String>
where
    Ops: OperatorSet<T = f32>,
{
    let mut nodes: Vec<PNode> = Vec::with_capacity(MAX_NODES);

    // Determine how many const slots are referenced.
    let mut max_const: i32 = -1;

    for &tok in packed.program.iter() {
        let kind = tok & 3;
        if kind == KIND_END {
            break;
        }
        let payload = tok >> 2;
        match kind {
            KIND_VAR => {
                nodes.push(PNode::Var {
                    feature: payload as u16,
                });
            }
            KIND_CONST => {
                let ci = payload as i32;
                if ci > max_const {
                    max_const = ci;
                }
                nodes.push(PNode::Const { idx: payload as u16 });
            }
            KIND_OP => {
                let arity = (payload >> 8) as u8;
                let opcode = (payload & 0xff) as u16;
                let name = gpu_opcode_to_name(arity, opcode)
                    .ok_or_else(|| format!("unsupported gpu opcode arity={arity} opcode={opcode}"))?;
                let op: OpId = Ops::lookup(name).map_err(|e: LookupError| format!("lookup({name}) failed: {e:?}"))?;
                nodes.push(PNode::Op {
                    arity: op.arity,
                    op: op.id,
                });
            }
            _ => {
                return Err(format!("unknown kind {kind}"));
            }
        }
    }

    let n_consts = if max_const < 0 {
        0
    } else {
        (max_const as usize + 1).min(MAX_CONSTS)
    };
    let consts = packed.consts[..n_consts].to_vec();

    Ok(PostfixExpr::new(nodes, consts, Metadata::default()))
}

/// Run a "full GPU" search.
///
/// This avoids the classic wgpu trap: *per-eval map/wait overhead*.
/// Only a single readback happens at the end.
pub fn equation_search_gpu_full<Ops, const D: usize>(
    dataset: &Dataset<f32>,
    options: &Options<f32, D>,
    cfg: GpuFullSearchConfig,
) -> Result<SearchResult<f32, Ops, D>, GpuInitError>
where
    Ops: OperatorSet<T = f32> + Send + Sync,
{
    if options.loss_kind != crate::loss_functions::LossKind::Mse {
        return Err(GpuInitError::UnsupportedLoss);
    }

    let pop_total = options.population_size.saturating_mul(options.populations.max(1));
    let pop_per_island = options.population_size.max(1);

    // Build masks for the operator set.
    let (allowed_unary_mask, allowed_binary_mask) = masks_from_options::<Ops, D>(options);

    let gpu = GpuFullSearcher::new(dataset, pop_total, pop_per_island)?;
    gpu.write_state_reset();

    let mutate_ppm = (cfg.mutation_rate.clamp(0.0, 1.0) * 1_000_000.0) as u32;
    let regen_ppm = (cfg.regen_rate.clamp(0.0, 1.0) * 1_000_000.0) as u32;

    // parity 0 = initial population
    let mut parity: u32 = 0;

    // init + eval
    let params0 = SearchParams {
        u0: [gpu.n_rows, gpu.n_features, gpu.pop_total, gpu.pop_per_island],
        u1: [0, parity, cfg.tournament_size.max(1), cfg.adam_iters],
        u2: [allowed_unary_mask, allowed_binary_mask, mutate_ppm, regen_ppm],
        u3: [cfg.seed, 0, 0, 0],
        f0: [gpu.sum_w, cfg.adam_lr, cfg.adam_beta1, cfg.adam_beta2],
        f1: [cfg.adam_eps, cfg.adam_step_clip, cfg.const_mut_sigma, 0.0],
    };
    gpu.write_params(&params0);
    gpu.dispatch(&gpu.pipeline_init, gpu.pop_total);

    // Evolve for `generations`. We keep everything on-GPU: no readbacks.
    for gen in 1..=cfg.generations {
        let params = SearchParams {
            u0: [gpu.n_rows, gpu.n_features, gpu.pop_total, gpu.pop_per_island],
            u1: [gen as u32, parity, cfg.tournament_size.max(1), cfg.adam_iters],
            u2: [allowed_unary_mask, allowed_binary_mask, mutate_ppm, regen_ppm],
            u3: [cfg.seed, 0, 0, 0],
            f0: [gpu.sum_w, cfg.adam_lr, cfg.adam_beta1, cfg.adam_beta2],
            f1: [cfg.adam_eps, cfg.adam_step_clip, cfg.const_mut_sigma, 0.0],
        };
        gpu.write_params(&params);
        gpu.dispatch(&gpu.pipeline_evolve, gpu.pop_total);
        parity ^= 1;
    }

    let (_best_loss_bits, best_index) = gpu.read_state_best();
    let packed_best = gpu.read_best_program(best_index);

    // Decode back into a PostfixExpr.
    let best_expr = unpack_program_to_expr::<Ops, D>(&packed_best).map_err(|e| {
        eprintln!("gpu full search: failed to unpack best program: {e}");
        GpuInitError::UnsupportedLoss
    })?;

    // Evaluate the found expression once on CPU to populate loss/cost/complexity using the normal path.
    let baseline_loss = if options.use_baseline {
        baseline_loss_from_zero_expression::<f32, Ops, D>(dataset, options.loss.as_ref())
    } else {
        None
    };
    let tagged = TaggedDataset::new(dataset, baseline_loss);

    let mut evaluator = Evaluator::<f32, D>::new(dataset.n_rows);
    let mut member = PopMember::from_expr(best_expr, dataset.n_features, options);
    let _ = member.evaluate(&tagged, options, &mut evaluator);

    let mut hof = HallOfFame::new(options.maxsize);
    hof.consider(&member, options, options.maxsize);

    Ok(SearchResult {
        hall_of_fame: hof,
        best: member,
    })
}
