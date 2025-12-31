// Symbolic Regression GPU kernels (WGSL)
//
// One workgroup per program p = workgroup_id.x.
// Threads stride over rows and reduce loss/grad inside the workgroup.
//
// Token encoding (matches Rust packer):
//   instr = (payload << 2) | kind
//   kind: 0 = VAR, 1 = CONST, 2 = OP, 3 = END
//   VAR payload: feature index
//   CONST payload: constant index (0..MAX_CONSTS-1)
//   OP payload: (arity << 8) | op_code
//
// MAX_NODES and MAX_CONSTS must match Rust constants in gpu/mod.rs.

const MAX_NODES: u32 = 32u;
const MAX_CONSTS: u32 = 8u;

// Tuning knob: smaller workgroups reduce wasted lanes when n_rows is small.
// Must be a power of two for the reductions.
const WG: u32 = 64u;

// Buffers
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> programs: array<u32>;
@group(0) @binding(4) var<storage, read_write> consts: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_loss: array<f32>;
// Extra output buffer:
//  - eval_mse_grad writes gradient here (p-major, MAX_CONSTS floats per program).
//  - optimize_adam doesn't use it.
@group(0) @binding(6) var<storage, read_write> out_extra: array<f32>;

struct Params {
  // u.x = n_rows
  // u.y = n_features
  // u.z = opt_iters
  // u.w = reserved
  u: vec4<u32>,

  // f0.x = sum_w
  // f0.y = opt_lr
  // f0.z = opt_beta1
  // f0.w = opt_beta2
  f0: vec4<f32>,

  // f1.x = opt_eps
  // f1.y = opt_step_clip
  // f1.zw reserved
  f1: vec4<f32>,
};

@group(0) @binding(7) var<uniform> params: Params;

fn op_unary_value(op: u32, a: f32) -> f32 {
  switch(op) {
    // 0: Neg
    case 0u: { return -a; }
    // 1: Sin
    case 1u: { return sin(a); }
    // 2: Cos
    case 2u: { return cos(a); }
    // 3: Exp
    case 3u: { return exp(a); }
    // 4: Log
    case 4u: { return log(a); }
    // 5: Sqrt
    case 5u: { return sqrt(a); }
    default: { return a; }
  }
}

fn op_unary_deriv(op: u32, a: f32, out: f32) -> f32 {
  switch(op) {
    // 0: Neg
    case 0u: { return -1.0; }
    // 1: Sin
    case 1u: { return cos(a); }
    // 2: Cos
    case 2u: { return -sin(a); }
    // 3: Exp
    case 3u: { return out; } // exp(a)
    // 4: Log
    case 4u: { return 1.0 / a; }
    // 5: Sqrt
    case 5u: { return 0.5 / out; } // 0.5 / sqrt(a)
    default: { return 0.0; }
  }
}

fn op_binary_value(op: u32, a: f32, b: f32) -> f32 {
  switch(op) {
    // 0: Add
    case 0u: { return a + b; }
    // 1: Sub
    case 1u: { return a - b; }
    // 2: Mul
    case 2u: { return a * b; }
    // 3: Div
    case 3u: { return a / b; }
    default: { return a; }
  }
}

// Returns (d_out/d_a, d_out/d_b)
fn op_binary_deriv(op: u32, a: f32, b: f32, out: f32) -> vec2<f32> {
  switch(op) {
    // 0: Add
    case 0u: { return vec2<f32>(1.0, 1.0); }
    // 1: Sub
    case 1u: { return vec2<f32>(1.0, -1.0); }
    // 2: Mul
    case 2u: { return vec2<f32>(b, a); }
    // 3: Div
    case 3u: { return vec2<f32>(1.0 / b, -a / (b * b)); }
    default: { return vec2<f32>(0.0, 0.0); }
  }
}

var<workgroup> wg_c0: vec4<f32>;
var<workgroup> wg_c1: vec4<f32>;
var<workgroup> wg_prog: array<u32, MAX_NODES>;


fn const_value(idx: u32) -> f32 {
  if (idx < 4u) {
    return wg_c0[i32(idx)];
  }
  return wg_c1[i32(idx - 4u)];
}

fn load_consts(p: u32) {
  let base = p * MAX_CONSTS;
  wg_c0 = vec4<f32>(
    consts[base + 0u],
    consts[base + 1u],
    consts[base + 2u],
    consts[base + 3u]
  );
  wg_c1 = vec4<f32>(
    consts[base + 4u],
    consts[base + 5u],
    consts[base + 6u],
    consts[base + 7u]
  );
}

fn load_shared(p: u32, lane: u32) {
  if (lane == 0u) {
    load_consts(p);
  }
  if (lane < MAX_NODES) {
    wg_prog[lane] = programs[p * MAX_NODES + lane];
  }
  workgroupBarrier();
}


fn store_consts(p: u32) {
  let base = p * MAX_CONSTS;
  consts[base + 0u] = wg_c0.x;
  consts[base + 1u] = wg_c0.y;
  consts[base + 2u] = wg_c0.z;
  consts[base + 3u] = wg_c0.w;

  consts[base + 4u] = wg_c1.x;
  consts[base + 5u] = wg_c1.y;
  consts[base + 6u] = wg_c1.z;
  consts[base + 7u] = wg_c1.w;
}

fn eval_postfix_value(p: u32, row: u32) -> f32 {
  let n_rows = params.u.x;
  let prog_off = p * MAX_NODES;

  var stack: array<f32, MAX_NODES>;
  var sp: i32 = 0;

  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    let instr = wg_prog[i];
    let kind = instr & 3u;
    if (kind == 3u) {
      break;
    }
    let payload = instr >> 2u;

    if (kind == 0u) {
      // VAR
      stack[sp] = X[payload * n_rows + row];
      sp = sp + 1;
    } else if (kind == 1u) {
      // CONST
      stack[sp] = const_value(payload);
      sp = sp + 1;
    } else {
      // OP
      let op_code = payload & 255u;
      let arity = payload >> 8u;
      if (arity == 1u) {
        sp = sp - 1;
        let a = stack[sp];
        stack[sp] = op_unary_value(op_code, a);
        sp = sp + 1;
      } else {
        sp = sp - 1;
        let b = stack[sp];
        sp = sp - 1;
        let a = stack[sp];
        stack[sp] = op_binary_value(op_code, a, b);
        sp = sp + 1;
      }
    }
  }

  return stack[sp - 1];
}

struct VGrad {
  v: f32,
  g0: vec4<f32>,
  g1: vec4<f32>,
};

// Reverse-mode AD to get d(output)/d(consts).
fn eval_postfix_value_and_grad(p: u32, row: u32) -> VGrad {
  let n_rows = params.u.x;
  let prog_off = p * MAX_NODES;

  // Per-node values and reverse adjoints
  var vals: array<f32, MAX_NODES>;
  var adj: array<f32, MAX_NODES>;

  // For OP nodes, record dependencies (node indices).
  var a_idx: array<u32, MAX_NODES>;
  var b_idx: array<u32, MAX_NODES>;

  // Stack holds node indices during forward pass.
  var stack_idx: array<u32, MAX_NODES>;
  var sp: i32 = 0;

  var n_nodes: u32 = 0u;

  // Forward pass
  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    let instr = wg_prog[i];
    let kind = instr & 3u;
    if (kind == 3u) {
      break;
    }
    n_nodes = i + 1u;
    adj[i] = 0.0;

    let payload = instr >> 2u;

    if (kind == 0u) {
      // VAR
      vals[i] = X[payload * n_rows + row];
      stack_idx[sp] = i;
      sp = sp + 1;
    } else if (kind == 1u) {
      // CONST
      vals[i] = const_value(payload);
      stack_idx[sp] = i;
      sp = sp + 1;
    } else {
      // OP
      let op_code = payload & 255u;
      let arity = payload >> 8u;

      if (arity == 1u) {
        sp = sp - 1;
        let a = stack_idx[sp];
        a_idx[i] = a;
        vals[i] = op_unary_value(op_code, vals[a]);
        stack_idx[sp] = i;
        sp = sp + 1;
      } else {
        sp = sp - 1;
        let b = stack_idx[sp];
        sp = sp - 1;
        let a = stack_idx[sp];
        a_idx[i] = a;
        b_idx[i] = b;
        vals[i] = op_binary_value(op_code, vals[a], vals[b]);
        stack_idx[sp] = i;
        sp = sp + 1;
      }
    }
  }

  let out_idx = stack_idx[sp - 1];
  adj[out_idx] = 1.0;

  var g0: vec4<f32> = vec4<f32>(0.0);
  var g1: vec4<f32> = vec4<f32>(0.0);

  // Reverse pass
  for (var k: u32 = 0u; k < n_nodes; k = k + 1u) {
    let i = (n_nodes - 1u) - k;
    let instr = wg_prog[i];
    let kind = instr & 3u;
    let payload = instr >> 2u;

    if (kind == 2u) {
      // OP
      let op_code = payload & 255u;
      let arity = payload >> 8u;
      let out_adj = adj[i];

      if (arity == 1u) {
        let a = a_idx[i];
        let da = op_unary_deriv(op_code, vals[a], vals[i]);
        adj[a] = adj[a] + out_adj * da;
      } else {
        let a = a_idx[i];
        let b = b_idx[i];
        let ders = op_binary_deriv(op_code, vals[a], vals[b], vals[i]);
        adj[a] = adj[a] + out_adj * ders.x;
        adj[b] = adj[b] + out_adj * ders.y;
      }
    } else if (kind == 1u) {
      // CONST
      let c = payload;
      if (c < 4u) {
        g0[i32(c)] = g0[i32(c)] + adj[i];
      } else if (c < 8u) {
        g1[i32(c - 4u)] = g1[i32(c - 4u)] + adj[i];
      }
    }
  }

  return VGrad(vals[out_idx], g0, g1);
}

var<workgroup> sh_loss: array<f32, WG>;
var<workgroup> sh_g0: array<vec4<f32>, WG>;
var<workgroup> sh_g1: array<vec4<f32>, WG>;

@compute @workgroup_size(WG, 1, 1)
fn eval_mse(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let p = gid.x;
  let lane = lid.x;

  load_shared(p, lane);
let n_rows = params.u.x;
  var sum: f32 = 0.0;

  for (var row: u32 = lane; row < n_rows; row = row + WG) {
    let pred = eval_postfix_value(p, row);
    let r = pred - y[row];
    let wi = w[row];
    sum = sum + wi * r * r;
  }

  sh_loss[lane] = sum;
  workgroupBarrier();

  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      sh_loss[lane] = sh_loss[lane] + sh_loss[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lane == 0u) {
    out_loss[p] = sh_loss[0] / params.f0.x; // sum_w
  }
}

@compute @workgroup_size(WG, 1, 1)
fn eval_mse_grad(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let p = gid.x;
  let lane = lid.x;

  load_shared(p, lane);
let n_rows = params.u.x;

  var sum: f32 = 0.0;
  var g0: vec4<f32> = vec4<f32>(0.0);
  var g1: vec4<f32> = vec4<f32>(0.0);

  for (var row: u32 = lane; row < n_rows; row = row + WG) {
    let out = eval_postfix_value_and_grad(p, row);
    let r = out.v - y[row];
    let wi = w[row];

    sum = sum + wi * r * r;

    let fac = 2.0 * wi * r;
    g0 = g0 + fac * out.g0;
    g1 = g1 + fac * out.g1;
  }

  sh_loss[lane] = sum;
  sh_g0[lane] = g0;
  sh_g1[lane] = g1;
  workgroupBarrier();

  var stride: u32 = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      sh_loss[lane] = sh_loss[lane] + sh_loss[lane + stride];
      sh_g0[lane] = sh_g0[lane] + sh_g0[lane + stride];
      sh_g1[lane] = sh_g1[lane] + sh_g1[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lane == 0u) {
    let inv_sw = 1.0 / params.f0.x; // 1/sum_w
    out_loss[p] = sh_loss[0] * inv_sw;

    let base = p * MAX_CONSTS;
    let gg0 = sh_g0[0] * inv_sw;
    let gg1 = sh_g1[0] * inv_sw;

    out_extra[base + 0u] = gg0.x;
    out_extra[base + 1u] = gg0.y;
    out_extra[base + 2u] = gg0.z;
    out_extra[base + 3u] = gg0.w;

    out_extra[base + 4u] = gg1.x;
    out_extra[base + 5u] = gg1.y;
    out_extra[base + 6u] = gg1.z;
    out_extra[base + 7u] = gg1.w;
  }
}

@compute @workgroup_size(WG, 1, 1)
fn optimize_adam(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let p = gid.x;
  let lane = lid.x;

  let iters = params.u.z;
  let sum_w = params.f0.x;

  let lr = params.f0.y;
  let beta1 = params.f0.z;
  let beta2 = params.f0.w;

  let eps = params.f1.x;
  let step_clip = params.f1.y;

  // Lane 0 owns parameter state (m/v and constants).
  var m0: vec4<f32> = vec4<f32>(0.0);
  var m1: vec4<f32> = vec4<f32>(0.0);
  var v0: vec4<f32> = vec4<f32>(0.0);
  var v1: vec4<f32> = vec4<f32>(0.0);
  var beta1_pow: f32 = 1.0;
  var beta2_pow: f32 = 1.0;
  var final_loss: f32 = 0.0;

  load_shared(p, lane);
for (var iter: u32 = 0u; iter < iters; iter = iter + 1u) {
    let n_rows = params.u.x;

    var sum: f32 = 0.0;
    var g0: vec4<f32> = vec4<f32>(0.0);
    var g1: vec4<f32> = vec4<f32>(0.0);

    for (var row: u32 = lane; row < n_rows; row = row + WG) {
      let out = eval_postfix_value_and_grad(p, row);
      let r = out.v - y[row];
      let wi = w[row];

      sum = sum + wi * r * r;

      let fac = 2.0 * wi * r;
      g0 = g0 + fac * out.g0;
      g1 = g1 + fac * out.g1;
    }

    sh_loss[lane] = sum;
    sh_g0[lane] = g0;
    sh_g1[lane] = g1;
    workgroupBarrier();

    // Reduce (loss, grad) across the workgroup.
    var stride: u32 = WG / 2u;
    loop {
      if (stride == 0u) { break; }
      if (lane < stride) {
        sh_loss[lane] = sh_loss[lane] + sh_loss[lane + stride];
        sh_g0[lane] = sh_g0[lane] + sh_g0[lane + stride];
        sh_g1[lane] = sh_g1[lane] + sh_g1[lane + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }

    if (lane == 0u) {
      let inv_sw = 1.0 / sum_w;
      final_loss = sh_loss[0] * inv_sw;

      let grad0 = sh_g0[0] * inv_sw;
      let grad1 = sh_g1[0] * inv_sw;

      // Adam update with cheap bias correction (no pow()).
      beta1_pow = beta1_pow * beta1;
      beta2_pow = beta2_pow * beta2;

      let one_minus_b1 = 1.0 - beta1;
      let one_minus_b2 = 1.0 - beta2;

      m0 = beta1 * m0 + one_minus_b1 * grad0;
      m1 = beta1 * m1 + one_minus_b1 * grad1;

      v0 = beta2 * v0 + one_minus_b2 * (grad0 * grad0);
      v1 = beta2 * v1 + one_minus_b2 * (grad1 * grad1);

      let bc1 = 1.0 - beta1_pow;
      let bc2 = 1.0 - beta2_pow;

      let m0_hat = m0 / bc1;
      let m1_hat = m1 / bc1;

      let v0_hat = v0 / bc2;
      let v1_hat = v1 / bc2;

      var step0 = lr * m0_hat / (sqrt(v0_hat) + vec4<f32>(eps));
      var step1 = lr * m1_hat / (sqrt(v1_hat) + vec4<f32>(eps));

      step0 = clamp(step0, vec4<f32>(-step_clip), vec4<f32>(step_clip));
      step1 = clamp(step1, vec4<f32>(-step_clip), vec4<f32>(step_clip));

      wg_c0 = wg_c0 - step0;
      wg_c1 = wg_c1 - step1;
    }

    workgroupBarrier();
  }

  if (lane == 0u) {
    store_consts(p);
    out_loss[p] = final_loss;
  }
}
