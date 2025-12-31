use std::thread;

use bytemuck::{Pod, Zeroable};
use crossbeam_channel as channel;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::{OpId, OperatorSet};
use num_traits::{Float, ToPrimitive};

use crate::Dataset;

pub const MAX_NODES: usize = 32;
pub const MAX_CONSTS: usize = 8;

const KERNELS_WGSL: &str = include_str!("shaders/kernels.wgsl");

const KIND_VAR: u32 = 0;
const KIND_CONST: u32 = 1;
const KIND_OP: u32 = 2;
const KIND_END: u32 = 3;
const END_TOKEN: u32 = KIND_END;

fn pack_var(feature: u16) -> u32 {
    KIND_VAR | ((feature as u32) << 2)
}

fn pack_const(idx: u16) -> u32 {
    KIND_CONST | ((idx as u32) << 2)
}

// New encoding: payload = (arity << 8) | opcode; instr = (payload << 2) | KIND_OP
fn pack_op(arity: u8, opcode: u16) -> u32 {
    let payload = ((arity as u32) << 8) | (opcode as u32);
    KIND_OP | (payload << 2)
}

#[repr(u16)]
#[derive(Copy, Clone, Debug)]
enum GpuBinaryOpcode {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
}

#[repr(u16)]
#[derive(Copy, Clone, Debug)]
enum GpuUnaryOpcode {
    Neg = 0,
    Sin = 1,
    Cos = 2,
    Exp = 3,
    Log = 4,
    Sqrt = 5,
}

fn map_op_to_gpu<Ops: OperatorSet>(op: OpId) -> Option<(u8, u16)> {
    let meta = Ops::meta(op)?;
    let arity = meta.arity;
    let opcode = match (meta.arity, meta.name) {
        (1, "neg") => GpuUnaryOpcode::Neg as u16,
        (1, "sin") => GpuUnaryOpcode::Sin as u16,
        (1, "cos") => GpuUnaryOpcode::Cos as u16,
        (1, "exp") => GpuUnaryOpcode::Exp as u16,
        (1, "log") => GpuUnaryOpcode::Log as u16,
        (1, "sqrt") => GpuUnaryOpcode::Sqrt as u16,

        (2, "add") => GpuBinaryOpcode::Add as u16,
        (2, "sub") => GpuBinaryOpcode::Sub as u16,
        (2, "mul") => GpuBinaryOpcode::Mul as u16,
        (2, "div") => GpuBinaryOpcode::Div as u16,
        _ => return None,
    };
    Some((arity, opcode))
}

#[derive(Copy, Clone, Debug)]
pub struct PackedProgram {
    pub program: [u32; MAX_NODES],
    pub consts: [f32; MAX_CONSTS],
}

#[derive(Copy, Clone, Debug)]
pub struct LossGrad {
    pub loss: f32,
    pub grad: [f32; MAX_CONSTS],
}

pub fn pack_expr<T, Ops, const D: usize>(expr: &PostfixExpr<T, Ops, D>) -> Option<PackedProgram>
where
    T: Float + ToPrimitive,
    Ops: OperatorSet<T = T>,
{
    if expr.nodes.len() > MAX_NODES {
        return None;
    }
    if expr.consts.len() > MAX_CONSTS {
        return None;
    }

    let mut program = [END_TOKEN; MAX_NODES];
    for (i, node) in expr.nodes.iter().enumerate() {
        let t = match *node {
            PNode::Var { feature } => pack_var(feature),
            PNode::Const { idx } => pack_const(idx),
            PNode::Op { arity, op } => {
                let (arity2, opcode) = map_op_to_gpu::<Ops>(OpId { arity, id: op })?;
                if arity2 != arity {
                    return None;
                }
                pack_op(arity, opcode)
            }
        };
        program[i] = t;
    }

    let mut consts = [0.0f32; MAX_CONSTS];
    for (i, c) in expr.consts.iter().copied().enumerate() {
        consts[i] = c.to_f32()?;
    }

    Some(PackedProgram { program, consts })
}

#[derive(Debug)]
pub enum GpuInitError {
    NoAdapter,
    NoCompute,
    RequestDeviceFailed,
}

/// Parameters for the fused constant optimizer (Adam) in `kernels.wgsl`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AdamParams {
    pub iters: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub step_clip: f32,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            iters: 64,
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step_clip: 0.25,
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    // u.x = n_rows
    // u.y = n_features
    // u.z = opt_iters
    // u.w = reserved
    u: [u32; 4],
    // f0.x = sum_w
    // f0.y = opt_lr
    // f0.z = opt_beta1
    // f0.w = opt_beta2
    f0: [f32; 4],
    // f1.x = opt_eps
    // f1.y = opt_step_clip
    // f1.zw reserved
    f1: [f32; 4],
}

struct GpuBatchEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline_mse: wgpu::ComputePipeline,
    pipeline_mse_grad: wgpu::ComputePipeline,
    pipeline_opt_adam: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // Dataset buffers
    _x_buf: wgpu::Buffer,
    _y_buf: wgpu::Buffer,
    _w_buf: wgpu::Buffer,

    // Inputs
    programs_buf: wgpu::Buffer,
    consts_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,

    // Outputs
    out_loss_buf: wgpu::Buffer,
    out_extra_buf: wgpu::Buffer,

    // Single readback buffer (loss + extra/consts)
    readback_buf: wgpu::Buffer,

    params: Params,
    p_max: usize,
}

impl GpuBatchEvaluator {
    fn new(dataset: &Dataset<f32>, p_max: usize) -> Result<Self, GpuInitError> {
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

        if std::env::var("SYMBOLIC_REGRESSION_GPU_PRINT_ADAPTER")
            .ok()
            .is_some_and(|v| v != "0")
        {
            let info = adapter.get_info();
            eprintln!(
                "symbolic_regression: GPU adapter: {} (backend={:?}, device_type={:?}, vendor=0x{:x}, device=0x{:x})",
                info.name, info.backend, info.device_type, info.vendor, info.device
            );
        }

        let downlevel = adapter.get_downlevel_capabilities();
        if !downlevel.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            return Err(GpuInitError::NoCompute);
        }

        // Metal often needs this explicitly to allow 7 storage buffers in compute.
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_storage_buffers_per_shader_stage = 8;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("symbolic_regression_gpu"),
            required_features: wgpu::Features::empty(),
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|_| GpuInitError::RequestDeviceFailed)?;

        let x_slice = dataset.x.as_slice().expect("dataset.x is contiguous");
        let y_slice = dataset.y_slice();
        let w_host: Vec<f32> = match dataset.weights_slice() {
            None => vec![1.0; dataset.n_rows],
            Some(w) => w.to_vec(),
        };
        let sum_w = w_host.iter().copied().sum::<f32>().max(1e-20);

        let params = Params {
            u: [dataset.n_rows as u32, dataset.n_features as u32, 0, 0],
            f0: [sum_w, 0.0, 0.0, 0.0],
            f1: [0.0, 0.0, 0.0, 0.0],
        };

        // Dataset buffers
        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x"),
            size: core::mem::size_of_val(x_slice) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x_slice));

        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: core::mem::size_of_val(y_slice) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&y_buf, 0, bytemuck::cast_slice(y_slice));

        let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w"),
            size: core::mem::size_of_val(w_host.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&w_buf, 0, bytemuck::cast_slice(&w_host));

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: core::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // Shared input buffers (capacity p_max)
        let programs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("programs"),
            size: (p_max * MAX_NODES * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let consts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("consts"),
            size: (p_max * MAX_CONSTS * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, // needed to read optimized constants back
            mapped_at_creation: false,
        });

        // Output buffers
        let out_loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_loss"),
            size: (p_max * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_extra_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_extra"),
            size: (p_max * MAX_CONSTS * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (p_max * (1 + MAX_CONSTS) * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shader & pipelines
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaders/kernels.wgsl"),
            source: wgpu::ShaderSource::Wgsl(KERNELS_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_eval_bind_group_layout"),
            entries: &[
                // 0: X
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
                // 1: y
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
                // 2: w
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
                // 3: programs
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: consts (read_write so optimize can update)
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
                // 5: out_loss
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
                // 6: out_extra
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
                // 7: params
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_eval_bind_group"),
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
                    resource: out_loss_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_extra_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_eval_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let shader_constants: [(&str, f64); 2] = [("MAX_NODES", MAX_NODES as f64), ("MAX_CONSTS", MAX_CONSTS as f64)];

        let create_pipeline = |label: &'static str, entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &shader_constants,
                    ..Default::default()
                },
                cache: None,
            })
        };

        let pipeline_mse = create_pipeline("gpu_eval_mse_pipeline", "eval_mse");
        let pipeline_mse_grad = create_pipeline("gpu_eval_mse_grad_pipeline", "eval_mse_grad");
        let pipeline_opt_adam = create_pipeline("gpu_eval_opt_adam_pipeline", "optimize_adam");

        Ok(Self {
            device,
            queue,
            pipeline_mse,
            pipeline_mse_grad,
            pipeline_opt_adam,
            bind_group,
            _x_buf: x_buf,
            _y_buf: y_buf,
            _w_buf: w_buf,
            programs_buf,
            consts_buf,
            params_buf,
            out_loss_buf,
            out_extra_buf,
            readback_buf,
            params,
            p_max,
        })
    }

    fn write_inputs(&mut self, programs: &[u32], consts: &[f32], p: usize) {
        assert!(p <= self.p_max);
        assert_eq!(programs.len(), p * MAX_NODES);
        assert_eq!(consts.len(), p * MAX_CONSTS);

        self.queue
            .write_buffer(&self.programs_buf, 0, bytemuck::cast_slice(programs));
        self.queue
            .write_buffer(&self.consts_buf, 0, bytemuck::cast_slice(consts));
    }

    fn write_params_base(&mut self) {
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    fn map_readback_f32(&self, n_f32: usize) -> Vec<f32> {
        let nbytes = (n_f32 * core::mem::size_of::<f32>()) as u64;
        let slice = self.readback_buf.slice(0..nbytes);

        let (tx, rx) = channel::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map callback runs").expect("map ok");

        let view = slice.get_mapped_range();
        let got: &[f32] = bytemuck::cast_slice(&view);
        let mut out = vec![0.0f32; n_f32];
        out.copy_from_slice(&got[..n_f32]);
        drop(view);
        self.readback_buf.unmap();
        out
    }

    fn eval_mse_batch(&mut self, programs: &[u32], consts: &[f32], p: usize, out_loss: &mut [f32]) {
        assert_eq!(out_loss.len(), p);

        self.write_inputs(programs, consts, p);

        // Params: only base (sum_w, n_rows)
        self.params.u[2] = 0;
        self.params.f0[1] = 0.0;
        self.params.f0[2] = 0.0;
        self.params.f0[3] = 0.0;
        self.params.f1 = [0.0; 4];
        self.write_params_base();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_eval_encoder_mse"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_eval_pass_mse"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_mse);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(p as u32, 1, 1);
        }

        let loss_bytes = (p * core::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.out_loss_buf, 0, &self.readback_buf, 0, loss_bytes);

        self.queue.submit([encoder.finish()]);

        let got = self.map_readback_f32(p);
        out_loss.copy_from_slice(&got);
    }

    fn eval_mse_grad_batch(
        &mut self,
        programs: &[u32],
        consts: &[f32],
        p: usize,
        out_loss: &mut [f32],
        out_grad: &mut [f32], // length p*MAX_CONSTS
    ) {
        assert_eq!(out_loss.len(), p);
        assert_eq!(out_grad.len(), p * MAX_CONSTS);

        self.write_inputs(programs, consts, p);

        // Params: only base
        self.params.u[2] = 0;
        self.params.f0[1] = 0.0;
        self.params.f0[2] = 0.0;
        self.params.f0[3] = 0.0;
        self.params.f1 = [0.0; 4];
        self.write_params_base();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_eval_encoder_mse_grad"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_eval_pass_mse_grad"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_mse_grad);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(p as u32, 1, 1);
        }

        let loss_bytes = (p * core::mem::size_of::<f32>()) as u64;
        let grad_bytes = (p * MAX_CONSTS * core::mem::size_of::<f32>()) as u64;
        let grad_off = loss_bytes;

        encoder.copy_buffer_to_buffer(&self.out_loss_buf, 0, &self.readback_buf, 0, loss_bytes);
        encoder.copy_buffer_to_buffer(&self.out_extra_buf, 0, &self.readback_buf, grad_off, grad_bytes);

        self.queue.submit([encoder.finish()]);

        let got = self.map_readback_f32(p * (1 + MAX_CONSTS));
        out_loss.copy_from_slice(&got[..p]);
        out_grad.copy_from_slice(&got[p..(p + p * MAX_CONSTS)]);
    }

    fn optimize_adam_batch(
        &mut self,
        programs: &[u32],
        consts: &[f32],
        p: usize,
        adam: AdamParams,
        out_loss: &mut [f32],
        out_consts: &mut [f32], // length p*MAX_CONSTS
    ) {
        assert_eq!(out_loss.len(), p);
        assert_eq!(out_consts.len(), p * MAX_CONSTS);

        self.write_inputs(programs, consts, p);

        self.params.u[2] = adam.iters;
        self.params.f0[1] = adam.lr;
        self.params.f0[2] = adam.beta1;
        self.params.f0[3] = adam.beta2;
        self.params.f1[0] = adam.eps;
        self.params.f1[1] = adam.step_clip;
        self.params.f1[2] = 0.0;
        self.params.f1[3] = 0.0;
        self.write_params_base();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_eval_encoder_opt_adam"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_eval_pass_opt_adam"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_opt_adam);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(p as u32, 1, 1);
        }

        let loss_bytes = (p * core::mem::size_of::<f32>()) as u64;
        let const_bytes = (p * MAX_CONSTS * core::mem::size_of::<f32>()) as u64;
        let const_off = loss_bytes;

        encoder.copy_buffer_to_buffer(&self.out_loss_buf, 0, &self.readback_buf, 0, loss_bytes);
        encoder.copy_buffer_to_buffer(&self.consts_buf, 0, &self.readback_buf, const_off, const_bytes);

        self.queue.submit([encoder.finish()]);

        let got = self.map_readback_f32(p * (1 + MAX_CONSTS));
        out_loss.copy_from_slice(&got[..p]);
        out_consts.copy_from_slice(&got[p..(p + p * MAX_CONSTS)]);
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum JobKind {
    Mse,
    MseGrad,
    OptimizeAdam(AdamParams),
}

enum JobResult {
    Mse(Vec<f32>),
    MseGrad { loss: Vec<f32>, grad: Vec<f32> },
    OptimizeAdam { loss: Vec<f32>, consts: Vec<f32> },
}

struct Job {
    kind: JobKind,
    p: usize,
    programs: Vec<u32>,
    consts: Vec<f32>,
    resp: channel::Sender<JobResult>,
}

fn gpu_server_loop(mut evaluator: GpuBatchEvaluator, rx: channel::Receiver<Job>) {
    let mut pending: Vec<Job> = Vec::new();

    // Scratch for merged jobs
    let mut jobs: Vec<Job> = Vec::with_capacity(evaluator.p_max);
    let mut programs: Vec<u32> = Vec::with_capacity(evaluator.p_max * MAX_NODES);
    let mut consts: Vec<f32> = Vec::with_capacity(evaluator.p_max * MAX_CONSTS);

    let mut out_loss: Vec<f32> = vec![0.0; evaluator.p_max];
    let mut out_extra: Vec<f32> = vec![0.0; evaluator.p_max * MAX_CONSTS];

    loop {
        jobs.clear();
        programs.clear();
        consts.clear();

        let first = if let Some(j) = pending.pop() {
            j
        } else {
            match rx.recv() {
                Ok(j) => j,
                Err(_) => return,
            }
        };

        let kind = first.kind;
        let mut total_p = 0usize;

        // push first
        total_p += first.p;
        jobs.push(first); // pull more, merge if same kind and capacity allows.
        // NOTE: On Metal (and sometimes Vulkan), the dominant per-dispatch cost is often the
        // "submit + map/readback wait". To amortize that, we opportunistically coalesce jobs
        // for a very short window after receiving the first request.
        let max_wait_us: u64 = std::env::var("SYMBOLIC_REGRESSION_GPU_BATCH_WAIT_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000);
        let mut min_fill: usize = std::env::var("SYMBOLIC_REGRESSION_GPU_BATCH_MIN_FILL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (evaluator.p_max / 2).clamp(64, 4096));
        min_fill = min_fill.min(evaluator.p_max).max(1);

        let deadline = std::time::Instant::now() + std::time::Duration::from_micros(max_wait_us);
        while total_p < evaluator.p_max {
            match rx.try_recv() {
                Ok(j) => {
                    if j.kind == kind && total_p + j.p <= evaluator.p_max {
                        total_p += j.p;
                        jobs.push(j);
                    } else {
                        pending.push(j);
                    }
                }
                Err(channel::TryRecvError::Empty) => {
                    // If we've already accumulated a reasonably large batch, dispatch immediately.
                    if total_p >= min_fill {
                        break;
                    }

                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }

                    // Block until either the next job arrives, or we hit the coalescing deadline.
                    let remaining = deadline.saturating_duration_since(now);
                    match rx.recv_timeout(remaining) {
                        Ok(j) => {
                            if j.kind == kind && total_p + j.p <= evaluator.p_max {
                                total_p += j.p;
                                jobs.push(j);
                            } else {
                                pending.push(j);
                            }
                        }
                        Err(channel::RecvTimeoutError::Timeout) => break,
                        Err(channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
                Err(channel::TryRecvError::Disconnected) => break,
            }
        }

        // Build merged inputs
        programs.reserve(total_p * MAX_NODES);
        consts.reserve(total_p * MAX_CONSTS);

        for j in &jobs {
            debug_assert_eq!(j.programs.len(), j.p * MAX_NODES);
            debug_assert_eq!(j.consts.len(), j.p * MAX_CONSTS);
            programs.extend_from_slice(&j.programs);
            consts.extend_from_slice(&j.consts);
        }

        match kind {
            JobKind::Mse => {
                evaluator.eval_mse_batch(&programs, &consts, total_p, &mut out_loss[..total_p]);

                let mut off = 0usize;
                for j in jobs.drain(..) {
                    let mut part = vec![0.0f32; j.p];
                    part.copy_from_slice(&out_loss[off..off + j.p]);
                    off += j.p;
                    let _ = j.resp.send(JobResult::Mse(part));
                }
            }
            JobKind::MseGrad => {
                evaluator.eval_mse_grad_batch(
                    &programs,
                    &consts,
                    total_p,
                    &mut out_loss[..total_p],
                    &mut out_extra[..(total_p * MAX_CONSTS)],
                );

                let mut off = 0usize;
                for j in jobs.drain(..) {
                    let mut part_loss = vec![0.0f32; j.p];
                    part_loss.copy_from_slice(&out_loss[off..off + j.p]);

                    let mut part_grad = vec![0.0f32; j.p * MAX_CONSTS];
                    let g0 = off * MAX_CONSTS;
                    part_grad.copy_from_slice(&out_extra[g0..g0 + j.p * MAX_CONSTS]);

                    off += j.p;
                    let _ = j.resp.send(JobResult::MseGrad {
                        loss: part_loss,
                        grad: part_grad,
                    });
                }
            }
            JobKind::OptimizeAdam(adam) => {
                evaluator.optimize_adam_batch(
                    &programs,
                    &consts,
                    total_p,
                    adam,
                    &mut out_loss[..total_p],
                    &mut out_extra[..(total_p * MAX_CONSTS)], // reuse buffer for consts
                );

                let mut off = 0usize;
                for j in jobs.drain(..) {
                    let mut part_loss = vec![0.0f32; j.p];
                    part_loss.copy_from_slice(&out_loss[off..off + j.p]);

                    let mut part_consts = vec![0.0f32; j.p * MAX_CONSTS];
                    let c0 = off * MAX_CONSTS;
                    part_consts.copy_from_slice(&out_extra[c0..c0 + j.p * MAX_CONSTS]);

                    off += j.p;
                    let _ = j.resp.send(JobResult::OptimizeAdam {
                        loss: part_loss,
                        consts: part_consts,
                    });
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct GpuClient {
    tx: channel::Sender<Job>,
    pub n_rows: usize,
    pub n_features: usize,
    pub p_max: usize,
}

impl GpuClient {
    pub fn spawn(dataset: &Dataset<f32>, batch_max: usize) -> Result<Self, GpuInitError> {
        let p_max = batch_max.max(1);
        let evaluator = GpuBatchEvaluator::new(dataset, p_max)?;

        let (tx, rx) = channel::unbounded::<Job>();
        let n_rows = dataset.n_rows;
        let n_features = dataset.n_features;

        thread::Builder::new()
            .name("symbolic_regression_gpu".to_string())
            .spawn(move || gpu_server_loop(evaluator, rx))
            .expect("spawn GPU server thread");

        Ok(Self {
            tx,
            n_rows,
            n_features,
            p_max,
        })
    }

    /// Maximum number of programs per GPU batch.
    pub fn max_batch(&self) -> usize {
        self.p_max
    }

    pub fn eval_mse(&self, program: PackedProgram) -> f32 {
        let mut out = [0.0f32; 1];
        self.eval_mse_many(&[program], &mut out);
        out[0]
    }

    /// Convenience single-program loss+grad.
    pub fn eval_mse_grad(&self, program: PackedProgram) -> LossGrad {
        let mut out_loss = [0.0f32; 1];
        let mut out_grad = [0.0f32; MAX_CONSTS];
        self.eval_mse_grad_many(&[program], &mut out_loss, &mut out_grad);
        LossGrad {
            loss: out_loss[0],
            grad: out_grad,
        }
    }

    /// Evaluate MSE for many programs (caller provides output slice).
    pub fn eval_mse_many(&self, programs: &[PackedProgram], out_loss: &mut [f32]) {
        assert_eq!(programs.len(), out_loss.len());

        for (chunk_idx, chunk) in programs.chunks(self.p_max).enumerate() {
            let out_chunk = &mut out_loss[chunk_idx * self.p_max..chunk_idx * self.p_max + chunk.len()];

            let mut prog_u32: Vec<u32> = Vec::with_capacity(chunk.len() * MAX_NODES);
            let mut const_f32: Vec<f32> = Vec::with_capacity(chunk.len() * MAX_CONSTS);
            for p in chunk {
                prog_u32.extend_from_slice(&p.program);
                const_f32.extend_from_slice(&p.consts);
            }

            let (tx, rx) = channel::bounded(1);
            let job = Job {
                kind: JobKind::Mse,
                p: chunk.len(),
                programs: prog_u32,
                consts: const_f32,
                resp: tx,
            };
            if self.tx.send(job).is_err() {
                for v in out_chunk.iter_mut() {
                    *v = f32::NAN;
                }
                return;
            }

            match rx.recv() {
                Ok(JobResult::Mse(loss)) => out_chunk.copy_from_slice(&loss),
                _ => {
                    for v in out_chunk.iter_mut() {
                        *v = f32::NAN;
                    }
                    return;
                }
            }
        }
    }

    /// Evaluate (loss, grad) for many programs. `out_grad` is flattened: p-major, MAX_CONSTS.
    pub fn eval_mse_grad_many(&self, programs: &[PackedProgram], out_loss: &mut [f32], out_grad: &mut [f32]) {
        assert_eq!(programs.len(), out_loss.len());
        assert_eq!(out_grad.len(), programs.len() * MAX_CONSTS);

        for (chunk_idx, chunk) in programs.chunks(self.p_max).enumerate() {
            let base = chunk_idx * self.p_max;
            let out_loss_chunk = &mut out_loss[base..base + chunk.len()];
            let out_grad_chunk = &mut out_grad[base * MAX_CONSTS..(base + chunk.len()) * MAX_CONSTS];

            let mut prog_u32: Vec<u32> = Vec::with_capacity(chunk.len() * MAX_NODES);
            let mut const_f32: Vec<f32> = Vec::with_capacity(chunk.len() * MAX_CONSTS);
            for p in chunk {
                prog_u32.extend_from_slice(&p.program);
                const_f32.extend_from_slice(&p.consts);
            }

            let (tx, rx) = channel::bounded(1);
            let job = Job {
                kind: JobKind::MseGrad,
                p: chunk.len(),
                programs: prog_u32,
                consts: const_f32,
                resp: tx,
            };
            if self.tx.send(job).is_err() {
                for v in out_loss_chunk.iter_mut() {
                    *v = f32::NAN;
                }
                for v in out_grad_chunk.iter_mut() {
                    *v = f32::NAN;
                }
                return;
            }

            match rx.recv() {
                Ok(JobResult::MseGrad { loss, grad }) => {
                    out_loss_chunk.copy_from_slice(&loss);
                    out_grad_chunk.copy_from_slice(&grad);
                }
                _ => {
                    for v in out_loss_chunk.iter_mut() {
                        *v = f32::NAN;
                    }
                    for v in out_grad_chunk.iter_mut() {
                        *v = f32::NAN;
                    }
                    return;
                }
            }
        }
    }

    /// Run the fused Adam optimizer on many programs in one or more GPU batches.
    ///
    /// - Input: `programs` slice contains initial constants.
    /// - Output: constants are updated in-place inside `programs`, and losses are written to `out_loss`.
    pub fn optimize_adam_many(&self, programs: &mut [PackedProgram], adam: AdamParams, out_loss: &mut [f32]) {
        assert_eq!(programs.len(), out_loss.len());

        for (chunk_idx, chunk) in programs.chunks_mut(self.p_max).enumerate() {
            let out_chunk = &mut out_loss[chunk_idx * self.p_max..chunk_idx * self.p_max + chunk.len()];

            let mut prog_u32: Vec<u32> = Vec::with_capacity(chunk.len() * MAX_NODES);
            let mut const_f32: Vec<f32> = Vec::with_capacity(chunk.len() * MAX_CONSTS);
            for p in chunk.iter() {
                prog_u32.extend_from_slice(&p.program);
                const_f32.extend_from_slice(&p.consts);
            }

            let (tx, rx) = channel::bounded(1);
            let job = Job {
                kind: JobKind::OptimizeAdam(adam),
                p: chunk.len(),
                programs: prog_u32,
                consts: const_f32,
                resp: tx,
            };

            if self.tx.send(job).is_err() {
                for v in out_chunk.iter_mut() {
                    *v = f32::NAN;
                }
                return;
            }

            match rx.recv() {
                Ok(JobResult::OptimizeAdam { loss, consts }) => {
                    out_chunk.copy_from_slice(&loss);
                    // write consts back into chunk programs
                    for (i, p) in chunk.iter_mut().enumerate() {
                        let base = i * MAX_CONSTS;
                        p.consts.copy_from_slice(&consts[base..base + MAX_CONSTS]);
                    }
                }
                _ => {
                    for v in out_chunk.iter_mut() {
                        *v = f32::NAN;
                    }
                    return;
                }
            }
        }
    }
}

#[cfg(all(test, feature = "gpu", not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn wgsl_parses() {
        let _ = naga::front::wgsl::parse_str(super::KERNELS_WGSL).expect("WGSL should parse");
    }
}
