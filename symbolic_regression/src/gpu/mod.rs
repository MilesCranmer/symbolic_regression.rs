use std::{thread, time};

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

impl core::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BatchingEnabled => write!(f, "GPU path does not support dataset batching"),
            Self::UnsupportedLoss => write!(f, "GPU path currently only supports MSE loss"),
            Self::NoAdapter => write!(
                f,
                "no compatible GPU adapter found (if running under a sandbox, try rerunning with full permissions; set SYMBOLIC_REGRESSION_GPU_DEBUG=1 to print adapters)"
            ),
            Self::NoCompute => write!(f, "GPU adapter does not support compute shaders"),
            Self::RequestDeviceFailed => write!(f, "failed to request GPU device"),
        }
    }
}

impl std::error::Error for GpuInitError {}

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

    _params_buf: wgpu::Buffer,

    p_max: usize,
}

impl GpuMseBatchEvaluator {
    fn new(dataset: &Dataset<f32>, p_max: usize) -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        if std::env::var("SYMBOLIC_REGRESSION_GPU_DEBUG")
            .ok()
            .is_some_and(|v| v != "0")
        {
            for (i, a) in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY))
                .into_iter()
                .enumerate()
            {
                let info = a.get_info();
                eprintln!(
                    "wgpu adapter[{i}]: name={} vendor={:#x} device={:#x} device_type={:?} backend={:?}",
                    info.name, info.vendor, info.device, info.device_type, info.backend
                );
            }
        }

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
            _params_buf: params_buf,
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

enum EvalRequest {
    Single {
        kind: EvalKind,
        program: [u32; MAX_NODES],
        consts: [f32; MAX_CONSTS],
        resp: channel::Sender<EvalResponse>,
    },
    BatchLoss {
        programs: Vec<u32>,
        consts: Vec<f32>,
        p: usize,
        resp: channel::Sender<Vec<f32>>,
    },
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
        let req = EvalRequest::Single {
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
        let req = EvalRequest::Single {
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

    pub fn eval_mse_many(&self, programs: &[PackedProgram], out: &mut [f32]) -> bool {
        assert_eq!(programs.len(), out.len());

        let p = programs.len();
        let mut packed_programs: Vec<u32> = Vec::with_capacity(p * MAX_NODES);
        let mut packed_consts: Vec<f32> = Vec::with_capacity(p * MAX_CONSTS);
        for program in programs {
            packed_programs.extend_from_slice(&program.program);
            packed_consts.extend_from_slice(&program.consts);
        }

        let (tx, rx) = channel::bounded(1);
        let req = EvalRequest::BatchLoss {
            programs: packed_programs,
            consts: packed_consts,
            p,
            resp: tx,
        };
        if self.tx.send(req).is_err() {
            return false;
        }

        let got = match rx.recv() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if got.len() != p {
            return false;
        }
        out.copy_from_slice(&got);
        true
    }
}

fn gpu_server_loop(mut evaluator: GpuMseBatchEvaluator, rx: channel::Receiver<EvalRequest>, batch_max: usize) {
    let batch_wait = time::Duration::from_micros(
        std::env::var("SYMBOLIC_REGRESSION_GPU_BATCH_WAIT_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(200),
    );
    let stats_enabled = std::env::var("SYMBOLIC_REGRESSION_GPU_STATS")
        .ok()
        .is_some_and(|v| v != "0");

    let mut pending: Vec<EvalRequest> = Vec::new();

    let mut reqs: Vec<EvalRequest> = Vec::with_capacity(batch_max);
    let mut programs: Vec<u32> = Vec::with_capacity(batch_max * MAX_NODES);
    let mut consts: Vec<f32> = Vec::with_capacity(batch_max * MAX_CONSTS);

    let mut out_loss: Vec<f32> = vec![0.0; batch_max];
    let mut out_grad: Vec<f32> = vec![0.0; batch_max * MAX_CONSTS];

    let mut last_report = time::Instant::now();
    let mut batches: u64 = 0;
    let mut total_programs: u64 = 0;
    let mut max_batch: usize = 0;
    let mut total_time = time::Duration::ZERO;
    let mut total_time_loss_only = time::Duration::ZERO;
    let mut total_time_loss_grad = time::Duration::ZERO;

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
        match first {
            EvalRequest::BatchLoss {
                programs,
                consts,
                p,
                resp,
            } => {
                if p == 0 {
                    let _ = resp.send(Vec::new());
                    continue;
                }
                if programs.len() != p * MAX_NODES || consts.len() != p * MAX_CONSTS {
                    let _ = resp.send(vec![f32::NAN; p]);
                    continue;
                }

                let t0 = time::Instant::now();
                let mut out = vec![0.0f32; p];
                let mut scratch = vec![0.0f32; batch_max];
                for chunk_start in (0..p).step_by(batch_max) {
                    let chunk_p = (p - chunk_start).min(batch_max);
                    let prog_start = chunk_start * MAX_NODES;
                    let prog_end = prog_start + chunk_p * MAX_NODES;
                    let const_start = chunk_start * MAX_CONSTS;
                    let const_end = const_start + chunk_p * MAX_CONSTS;

                    evaluator.eval_mse_batch(
                        &programs[prog_start..prog_end],
                        &consts[const_start..const_end],
                        chunk_p,
                        &mut scratch[..chunk_p],
                    );
                    out[chunk_start..(chunk_start + chunk_p)].copy_from_slice(&scratch[..chunk_p]);
                }
                let _ = resp.send(out);

                let dt = t0.elapsed();
                batches += 1;
                total_programs += p as u64;
                max_batch = max_batch.max(p);
                total_time += dt;
                total_time_loss_only += dt;

                if stats_enabled && last_report.elapsed() >= time::Duration::from_secs(2) {
                    let avg_batch = (total_programs as f64) / (batches as f64).max(1.0);
                    eprintln!(
                        "gpu server: batches={batches} programs={total_programs} avg_batch={avg_batch:.1} max_batch={max_batch} total_time_ms={} loss_only_ms={} loss_grad_ms={} batch_wait_us={}",
                        total_time.as_millis(),
                        total_time_loss_only.as_millis(),
                        total_time_loss_grad.as_millis(),
                        batch_wait.as_micros(),
                    );
                    last_report = time::Instant::now();
                }
                continue;
            }
            EvalRequest::Single { .. } => {
                reqs.push(first);
            }
        }

        let kind = match reqs[0] {
            EvalRequest::Single { kind, .. } => kind,
            EvalRequest::BatchLoss { .. } => unreachable!("batch handled above"),
        };
        let deadline = time::Instant::now().checked_add(batch_wait);
        while reqs.len() < batch_max {
            loop {
                match rx.try_recv() {
                    Ok(r) => match r {
                        EvalRequest::Single { kind: k, .. } if k == kind => reqs.push(r),
                        _ => pending.push(r),
                    },
                    Err(channel::TryRecvError::Empty) => break,
                    Err(channel::TryRecvError::Disconnected) => break,
                }
                if reqs.len() >= batch_max {
                    break;
                }
            }

            let Some(deadline) = deadline else {
                break;
            };
            let now = time::Instant::now();
            if now >= deadline {
                break;
            }

            match rx.recv_timeout(deadline.saturating_duration_since(now)) {
                Ok(r) => match r {
                    EvalRequest::Single { kind: k, .. } if k == kind => reqs.push(r),
                    _ => pending.push(r),
                },
                Err(channel::RecvTimeoutError::Timeout) => break,
                Err(channel::RecvTimeoutError::Disconnected) => break,
            }
        }

        let p = reqs.len();
        batches += 1;
        total_programs += p as u64;
        max_batch = max_batch.max(p);
        programs.reserve(p * MAX_NODES);
        consts.reserve(p * MAX_CONSTS);
        for r in &reqs {
            let (program, consts_i) = match r {
                EvalRequest::Single { program, consts, .. } => (program, consts),
                EvalRequest::BatchLoss { .. } => unreachable!("only single reqs in this vec"),
            };
            programs.extend_from_slice(program);
            consts.extend_from_slice(consts_i);
        }

        let t0 = time::Instant::now();
        match kind {
            EvalKind::LossOnly => {
                evaluator.eval_mse_batch(&programs, &consts, p, &mut out_loss[..p]);
                for (i, r) in reqs.drain(..).enumerate() {
                    let resp = match r {
                        EvalRequest::Single { resp, .. } => resp,
                        EvalRequest::BatchLoss { .. } => unreachable!("only single reqs in this vec"),
                    };
                    let _ = resp.send(EvalResponse::Loss(out_loss[i]));
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
                    let resp = match r {
                        EvalRequest::Single { resp, .. } => resp,
                        EvalRequest::BatchLoss { .. } => unreachable!("only single reqs in this vec"),
                    };
                    let _ = resp.send(EvalResponse::LossGrad(LossGrad {
                        loss: out_loss[i],
                        grad: g,
                    }));
                }
            }
        }

        let dt = t0.elapsed();
        total_time += dt;
        match kind {
            EvalKind::LossOnly => total_time_loss_only += dt,
            EvalKind::LossGrad => total_time_loss_grad += dt,
        }

        if stats_enabled && last_report.elapsed() >= time::Duration::from_secs(2) {
            let avg_batch = (total_programs as f64) / (batches as f64).max(1.0);
            eprintln!(
                "gpu server: batches={batches} programs={total_programs} avg_batch={avg_batch:.1} max_batch={max_batch} total_time_ms={} loss_only_ms={} loss_grad_ms={} batch_wait_us={}",
                total_time.as_millis(),
                total_time_loss_only.as_millis(),
                total_time_loss_grad.as_millis(),
                batch_wait.as_micros(),
            );
            last_report = time::Instant::now();
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
