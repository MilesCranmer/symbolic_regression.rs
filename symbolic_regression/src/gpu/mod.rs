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

const KERNELS_WGSL: &str = include_str!("kernels.wgsl");

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

fn pack_op(arity: u8, opcode: u16) -> u32 {
    KIND_OP | ((arity as u32) << 2) | ((opcode as u32) << 10)
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
    BatchingEnabled,
    UnsupportedLoss,
    NoAdapter,
    NoCompute,
    RequestDeviceFailed,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n_rows: u32,
    n_features: u32,
    sum_w: f32,
    _pad0: u32,
}

struct GpuMseBatchEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pipeline_mse: wgpu::ComputePipeline,
    bind_group_mse: wgpu::BindGroup,

    pipeline_mse_grad: wgpu::ComputePipeline,
    bind_group_mse_grad: wgpu::BindGroup,

    programs_buf: wgpu::Buffer,
    consts_buf: wgpu::Buffer,
    out_loss_buf: wgpu::Buffer,
    readback_loss_buf: wgpu::Buffer,

    out_grad_buf: wgpu::Buffer,
    readback_grad_buf: wgpu::Buffer,

    params: Params,
    params_buf: wgpu::Buffer,

    p_max: usize,
}

impl GpuMseBatchEvaluator {
    fn new(dataset: &Dataset<f32>, p_max: usize) -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| GpuInitError::NoAdapter)?;

        let downlevel = adapter.get_downlevel_capabilities();
        if !downlevel.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            return Err(GpuInitError::NoCompute);
        }

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
        let sum_w = w_host.iter().copied().sum::<f32>();
        let params = Params {
            n_rows: dataset.n_rows as u32,
            n_features: dataset.n_features as u32,
            sum_w,
            _pad0: 0,
        };

        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x"),
            size: (x_slice.len() * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(x_slice));

        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: (y_slice.len() * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&y_buf, 0, bytemuck::cast_slice(y_slice));

        let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w"),
            size: (w_host.len() * core::mem::size_of::<f32>()) as u64,
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

        let programs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("programs"),
            size: (p_max * MAX_NODES * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let consts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("consts"),
            size: (p_max * MAX_CONSTS * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_loss"),
            size: (p_max * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_loss"),
            size: (p_max * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_grad"),
            size: (p_max * MAX_CONSTS * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_grad"),
            size: (p_max * MAX_CONSTS * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_kernels.wgsl"),
            source: wgpu::ShaderSource::Wgsl(KERNELS_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_eval_bind_group_layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let bind_group_mse = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_eval_mse_bind_group"),
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
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_eval_mse_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline_mse = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_eval_mse_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("eval_mse"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout_grad = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_eval_grad_bind_group_layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_mse_grad = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_eval_grad_bind_group"),
            layout: &bind_group_layout_grad,
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
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: out_grad_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_grad = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_eval_grad_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout_grad],
            immediate_size: 0,
        });

        let pipeline_mse_grad = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_eval_grad_pipeline"),
            layout: Some(&pipeline_layout_grad),
            module: &shader,
            entry_point: Some("eval_mse_grad"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline_mse,
            bind_group_mse,
            pipeline_mse_grad,
            bind_group_mse_grad,
            programs_buf,
            consts_buf,
            out_loss_buf,
            readback_loss_buf,
            out_grad_buf,
            readback_grad_buf,
            params,
            params_buf,
            p_max,
        })
    }

    fn eval_mse_batch(&mut self, programs: &[u32], consts: &[f32], p: usize, out: &mut [f32]) {
        assert!(p <= self.p_max);
        assert_eq!(programs.len(), p * MAX_NODES);
        assert_eq!(consts.len(), p * MAX_CONSTS);
        assert_eq!(out.len(), p);

        self.queue
            .write_buffer(&self.programs_buf, 0, bytemuck::cast_slice(programs));
        self.queue
            .write_buffer(&self.consts_buf, 0, bytemuck::cast_slice(consts));
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_eval_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_eval_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_mse);
            pass.set_bind_group(0, &self.bind_group_mse, &[]);
            pass.dispatch_workgroups(p as u32, 1, 1);
        }

        let nbytes = (p * core::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.out_loss_buf, 0, &self.readback_loss_buf, 0, nbytes);

        self.queue.submit([encoder.finish()]);

        let slice = self.readback_loss_buf.slice(0..nbytes);
        let (tx, rx) = channel::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map callback runs").expect("map ok");

        let view = slice.get_mapped_range();
        let got: &[f32] = bytemuck::cast_slice(&view);
        out.copy_from_slice(got);
        drop(view);
        self.readback_loss_buf.unmap();
    }

    fn eval_mse_grad_batch(
        &mut self,
        programs: &[u32],
        consts: &[f32],
        p: usize,
        out_loss: &mut [f32],
        out_grad: &mut [f32],
    ) {
        assert!(p <= self.p_max);
        assert_eq!(programs.len(), p * MAX_NODES);
        assert_eq!(consts.len(), p * MAX_CONSTS);
        assert_eq!(out_loss.len(), p);
        assert_eq!(out_grad.len(), p * MAX_CONSTS);

        self.queue
            .write_buffer(&self.programs_buf, 0, bytemuck::cast_slice(programs));
        self.queue
            .write_buffer(&self.consts_buf, 0, bytemuck::cast_slice(consts));
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_eval_grad_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_eval_grad_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_mse_grad);
            pass.set_bind_group(0, &self.bind_group_mse_grad, &[]);
            pass.dispatch_workgroups(p as u32, 1, 1);
        }

        let loss_bytes = (p * core::mem::size_of::<f32>()) as u64;
        let grad_bytes = (p * MAX_CONSTS * core::mem::size_of::<f32>()) as u64;

        encoder.copy_buffer_to_buffer(&self.out_loss_buf, 0, &self.readback_loss_buf, 0, loss_bytes);
        encoder.copy_buffer_to_buffer(&self.out_grad_buf, 0, &self.readback_grad_buf, 0, grad_bytes);

        self.queue.submit([encoder.finish()]);

        let loss_slice = self.readback_loss_buf.slice(0..loss_bytes);
        let grad_slice = self.readback_grad_buf.slice(0..grad_bytes);

        let (tx, rx) = channel::bounded(2);

        loss_slice.map_async(wgpu::MapMode::Read, {
            let tx = tx.clone();
            move |res| {
                let _ = tx.send(res);
            }
        });
        grad_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        for _ in 0..2 {
            rx.recv().expect("map callback runs").expect("map ok");
        }

        {
            let view = loss_slice.get_mapped_range();
            let got: &[f32] = bytemuck::cast_slice(&view);
            out_loss.copy_from_slice(&got[..p]);
            drop(view);
            self.readback_loss_buf.unmap();
        }
        {
            let view = grad_slice.get_mapped_range();
            let got: &[f32] = bytemuck::cast_slice(&view);
            out_grad.copy_from_slice(&got[..(p * MAX_CONSTS)]);
            drop(view);
            self.readback_grad_buf.unmap();
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EvalKind {
    LossOnly,
    LossGrad,
}

#[derive(Copy, Clone, Debug)]
enum EvalResponse {
    Loss(f32),
    LossGrad(LossGrad),
}

struct EvalRequest {
    kind: EvalKind,
    program: [u32; MAX_NODES],
    consts: [f32; MAX_CONSTS],
    resp: channel::Sender<EvalResponse>,
}

#[derive(Clone)]
pub struct GpuClient {
    tx: channel::Sender<EvalRequest>,
    pub n_rows: usize,
    pub n_features: usize,
}

impl GpuClient {
    pub fn spawn(dataset: &Dataset<f32>, batch_max: usize) -> Result<Self, GpuInitError> {
        let p_max = batch_max.max(1);
        let evaluator = GpuMseBatchEvaluator::new(dataset, p_max)?;

        let (tx, rx) = channel::unbounded::<EvalRequest>();
        let n_rows = dataset.n_rows;
        let n_features = dataset.n_features;

        thread::Builder::new()
            .name("symbolic_regression_gpu".to_string())
            .spawn(move || gpu_server_loop(evaluator, rx, p_max))
            .expect("spawn GPU server thread");

        Ok(Self { tx, n_rows, n_features })
    }

    pub fn eval_mse(&self, program: PackedProgram) -> f32 {
        let (tx, rx) = channel::bounded(1);
        let req = EvalRequest {
            kind: EvalKind::LossOnly,
            program: program.program,
            consts: program.consts,
            resp: tx,
        };
        if self.tx.send(req).is_err() {
            return f32::NAN;
        }
        match rx.recv().ok() {
            Some(EvalResponse::Loss(v)) => v,
            _ => f32::NAN,
        }
    }

    pub fn eval_mse_grad(&self, program: PackedProgram) -> LossGrad {
        let (tx, rx) = channel::bounded(1);
        let req = EvalRequest {
            kind: EvalKind::LossGrad,
            program: program.program,
            consts: program.consts,
            resp: tx,
        };
        if self.tx.send(req).is_err() {
            return LossGrad {
                loss: f32::NAN,
                grad: [0.0; MAX_CONSTS],
            };
        }
        match rx.recv().ok() {
            Some(EvalResponse::LossGrad(v)) => v,
            _ => LossGrad {
                loss: f32::NAN,
                grad: [0.0; MAX_CONSTS],
            },
        }
    }
}

fn gpu_server_loop(mut evaluator: GpuMseBatchEvaluator, rx: channel::Receiver<EvalRequest>, batch_max: usize) {
    let mut pending: Vec<EvalRequest> = Vec::new();

    let mut reqs: Vec<EvalRequest> = Vec::with_capacity(batch_max);
    let mut programs: Vec<u32> = Vec::with_capacity(batch_max * MAX_NODES);
    let mut consts: Vec<f32> = Vec::with_capacity(batch_max * MAX_CONSTS);

    let mut out_loss: Vec<f32> = vec![0.0; batch_max];
    let mut out_grad: Vec<f32> = vec![0.0; batch_max * MAX_CONSTS];

    loop {
        reqs.clear();
        programs.clear();
        consts.clear();

        let first = if let Some(r) = pending.pop() {
            r
        } else {
            match rx.recv() {
                Ok(r) => r,
                Err(_) => return,
            }
        };
        reqs.push(first);

        let kind = reqs[0].kind;
        while reqs.len() < batch_max {
            match rx.try_recv() {
                Ok(r) => {
                    if r.kind == kind {
                        reqs.push(r);
                    } else {
                        pending.push(r);
                    }
                }
                Err(channel::TryRecvError::Empty) => break,
                Err(channel::TryRecvError::Disconnected) => break,
            }
        }

        let p = reqs.len();
        programs.reserve(p * MAX_NODES);
        consts.reserve(p * MAX_CONSTS);
        for r in &reqs {
            programs.extend_from_slice(&r.program);
            consts.extend_from_slice(&r.consts);
        }

        match kind {
            EvalKind::LossOnly => {
                evaluator.eval_mse_batch(&programs, &consts, p, &mut out_loss[..p]);
                for (i, r) in reqs.drain(..).enumerate() {
                    let _ = r.resp.send(EvalResponse::Loss(out_loss[i]));
                }
            }
            EvalKind::LossGrad => {
                evaluator.eval_mse_grad_batch(
                    &programs,
                    &consts,
                    p,
                    &mut out_loss[..p],
                    &mut out_grad[..(p * MAX_CONSTS)],
                );
                for (i, r) in reqs.drain(..).enumerate() {
                    let mut g = [0.0f32; MAX_CONSTS];
                    let base = i * MAX_CONSTS;
                    g.copy_from_slice(&out_grad[base..base + MAX_CONSTS]);
                    let _ = r.resp.send(EvalResponse::LossGrad(LossGrad {
                        loss: out_loss[i],
                        grad: g,
                    }));
                }
            }
        }
    }
}

#[cfg(all(test, feature = "gpu", not(target_arch = "wasm32")))]
mod tests {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::expression::{Metadata, PostfixExpr};
    use dynamic_expressions::node::PNode;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::{GpuClient, pack_expr};
    use crate::Dataset;

    fn extract_u32_const(src: &str, name: &str) -> Option<u32> {
        let pat = format!("const {name}:");
        let start = src.find(&pat)?;
        let after = &src[(start + pat.len())..];
        let eq = after.find('=')?;
        let after_eq = after[(eq + 1)..].trim_start();

        let mut digits = String::new();
        for ch in after_eq.chars() {
            if ch.is_ascii_digit() {
                digits.push(ch);
                continue;
            }
            if ch.is_whitespace() {
                continue;
            }
            break;
        }

        digits.parse().ok()
    }

    #[test]
    fn wgsl_constants_match_rust() {
        assert_eq!(
            extract_u32_const(super::KERNELS_WGSL, "MAX_NODES"),
            Some(super::MAX_NODES as u32)
        );
        assert_eq!(
            extract_u32_const(super::KERNELS_WGSL, "MAX_CONSTS"),
            Some(super::MAX_CONSTS as u32)
        );
    }

    #[test]
    #[ignore = "requires a working native GPU backend (Metal/Vulkan); run with `cargo test -p symbolic_regression --features gpu --lib -- --ignored --nocapture`"]
    fn gpu_mse_matches_cpu_for_simple_add_expr() {
        let mut rng = StdRng::seed_from_u64(0);

        let n_rows = 512;
        let n_features = 2;
        let mut x = Array2::<f32>::zeros((n_features, n_rows));
        let mut y = Array1::<f32>::zeros(n_rows);

        for row in 0..n_rows {
            let x0 = rng.random_range(-1.0f32..1.0f32);
            let x1 = rng.random_range(-1.0f32..1.0f32);
            x[(0, row)] = x0;
            x[(1, row)] = x1;
            y[row] = x0 + x1;
        }

        let dataset = Dataset::new(x, y);

        let add = BuiltinOpsF32::lookup("+").expect("builtin has +");
        let expr = PostfixExpr::<f32, BuiltinOpsF32, 3>::new(
            vec![
                PNode::Var { feature: 0 },
                PNode::Var { feature: 1 },
                PNode::Op {
                    arity: add.arity,
                    op: add.id,
                },
            ],
            Vec::new(),
            Metadata::default(),
        );

        let plan = dynamic_expressions::compile_plan(&expr.nodes, dataset.n_features, expr.consts.len());
        let mut yhat = vec![0.0f32; dataset.n_rows];
        let mut scratch = ndarray::Array2::<f32>::zeros((0, 0));
        let eval_opts = dynamic_expressions::EvalOptions {
            check_finite: true,
            early_exit: true,
        };
        let ok = dynamic_expressions::eval_plan_array_into(
            &mut yhat,
            &plan,
            &expr,
            dataset.x.view(),
            &mut scratch,
            &eval_opts,
        );
        assert!(ok, "cpu eval failed");

        let mut sum = 0.0f32;
        for (a, b) in yhat.iter().copied().zip(dataset.y_slice().iter().copied()) {
            let r = a - b;
            sum += r * r;
        }
        let cpu_mse = sum / (dataset.n_rows as f32);

        let gpu = GpuClient::spawn(&dataset, 256).expect("gpu init failed");
        let packed = pack_expr(&expr).expect("expr should be gpu-packable");
        let gpu_mse = gpu.eval_mse(packed);

        let rel = (gpu_mse - cpu_mse).abs() / cpu_mse.max(1e-8);
        assert!(rel < 1e-5, "cpu_mse={cpu_mse} gpu_mse={gpu_mse} rel={rel}");
    }

    #[test]
    #[ignore = "requires a working native GPU backend (Metal/Vulkan); run with `cargo test -p symbolic_regression --features gpu --lib -- --ignored --nocapture`"]
    fn gpu_mse_grad_matches_known_value_for_linear_const() {
        let n_rows = 512;
        let n_features = 1;
        let mut x = Array2::<f32>::zeros((n_features, n_rows));
        let mut y = Array1::<f32>::zeros(n_rows);

        // y = x0, so for yhat = x0 + c0, residual is constant r=c0.
        for row in 0..n_rows {
            let v = row as f32 * 0.01;
            x[(0, row)] = v;
            y[row] = v;
        }

        let dataset = Dataset::new(x, y);

        let add = BuiltinOpsF32::lookup("+").expect("builtin has +");
        let c0 = 0.123f32;
        let expr = PostfixExpr::<f32, BuiltinOpsF32, 3>::new(
            vec![
                PNode::Var { feature: 0 },
                PNode::Const { idx: 0 },
                PNode::Op {
                    arity: add.arity,
                    op: add.id,
                },
            ],
            vec![c0],
            Metadata::default(),
        );

        let packed = pack_expr(&expr).expect("expr should be gpu-packable");
        let gpu = GpuClient::spawn(&dataset, 256).expect("gpu init failed");
        let res = gpu.eval_mse_grad(packed);

        let expected_loss = c0 * c0;
        let expected_grad0 = 2.0 * c0;

        let loss_err = (res.loss - expected_loss).abs();
        let grad_err = (res.grad[0] - expected_grad0).abs();
        assert!(
            loss_err < 1e-5 && grad_err < 1e-5,
            "loss={}, grad0={}, expected_loss={}, expected_grad0={}, loss_err={}, grad_err={}",
            res.loss,
            res.grad[0],
            expected_loss,
            expected_grad0,
            loss_err,
            grad_err
        );
    }

    #[test]
    fn gpu_mse_grad_smoke_or_skip_when_no_adapter() {
        let n_rows = 64;
        let n_features = 1;
        let mut x = Array2::<f32>::zeros((n_features, n_rows));
        let mut y = Array1::<f32>::zeros(n_rows);

        for row in 0..n_rows {
            let v = row as f32 * 0.01;
            x[(0, row)] = v;
            y[row] = v;
        }

        let dataset = Dataset::new(x, y);

        let gpu = match GpuClient::spawn(&dataset, 32) {
            Ok(gpu) => gpu,
            Err(_) => return,
        };

        let add = BuiltinOpsF32::lookup("+").expect("builtin has +");
        let c0 = 0.25f32;
        let expr = PostfixExpr::<f32, BuiltinOpsF32, 3>::new(
            vec![
                PNode::Var { feature: 0 },
                PNode::Const { idx: 0 },
                PNode::Op {
                    arity: add.arity,
                    op: add.id,
                },
            ],
            vec![c0],
            Metadata::default(),
        );
        let packed = pack_expr(&expr).expect("expr should be gpu-packable");
        let res = gpu.eval_mse_grad(packed);

        let expected_loss = c0 * c0;
        let expected_grad0 = 2.0 * c0;
        assert!((res.loss - expected_loss).abs() < 1e-4);
        assert!((res.grad[0] - expected_grad0).abs() < 1e-4);
    }
}
