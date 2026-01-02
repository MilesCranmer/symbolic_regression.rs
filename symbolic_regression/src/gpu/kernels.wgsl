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
//  - optimize_lm_grid doesn't use it.
@group(0) @binding(6) var<storage, read_write> out_extra: array<f32>;

struct Params {
  // u.x = n_rows
  // u.y = n_features (currently unused; dataset is column-major in X)
  // u.z = opt_steps (LM outer steps)
  // u.w = reserved
  u: vec4<u32>,

  // f0.x = sum_w
  // f0.y = lm_lambda_scale  (lambda_base = max(lm_lambda_floor, lm_lambda_scale * mean(diag(J^T W J))))
  // f0.z = lm_lambda_floor
  // f0.w = lm_step_clip     (clamp per-parameter step to [-clip, +clip]; 0 => no clamp)
  f0: vec4<f32>,

  // f1 reserved for future kernels / tuning knobs.
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
var<workgroup> wg_done: u32;
var<workgroup> wg_prog: array<u32, MAX_NODES>;


fn const_value(idx: u32) -> f32 {
  if (idx < 4u) {
    return wg_c0[i32(idx)];
  }
  return wg_c1[i32(idx - 4u)];
}
fn const_value_from(c0: vec4<f32>, c1: vec4<f32>, idx: u32) -> f32 {
  if (idx < 4u) {
    return c0[i32(idx)];
  }
  return c1[i32(idx - 4u)];
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
fn eval_postfix_value_with_consts(row: u32, c0: vec4<f32>, c1: vec4<f32>) -> f32 {
  let n_rows = params.u.x;

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
      stack[sp] = const_value_from(c0, c1, payload);
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

// --- LM optimizer workgroup state ---
// Adaptive damping LM / damped Gauss-Newton.
// We avoid evaluating a full lambda grid each step (which multiplies forward-eval cost).
//
// Per LM step, we compute SSE + g + H, then try up to LM_MAX_TRIES damping values
// sequentially (each try is one forward SSE pass). This keeps the kernel work closer
// to ~1x (value+grad) + ~tries*(value), instead of (value+grad) + 7*(value).
const LM_MAX_TRIES: u32 = 2u;

var<workgroup> sh_h00: array<mat4x4<f32>, WG>;
var<workgroup> sh_h01: array<mat4x4<f32>, WG>;
var<workgroup> sh_h11: array<mat4x4<f32>, WG>;

var<workgroup> wg_base_sse: f32;
var<workgroup> wg_lambda: f32;
var<workgroup> wg_trial_c0: vec4<f32>;
var<workgroup> wg_trial_c1: vec4<f32>;
var<workgroup> wg_trial_sse: f32;
var<workgroup> wg_accept: u32;

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


fn outer4(a: vec4<f32>, b: vec4<f32>) -> mat4x4<f32> {
  // Column-major matrix: column j is a * b[j]
  return mat4x4<f32>(
    a * b.x,
    a * b.y,
    a * b.z,
    a * b.w
  );
}

fn is_finite(x: f32) -> bool {
  // NaN check: NaN != NaN. Inf check: abs(inf) is inf.
  return (x == x) && (abs(x) < 3.0e38);
}

struct Solve8Res {
  ok: bool,
  dx0: vec4<f32>,
  dx1: vec4<f32>,
};

fn solve8(A_in: array<f32, 64>, b_in: array<f32, 8>) -> Solve8Res {
  var A = A_in;
  var b = b_in;

  // Forward elimination with partial pivoting.
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    // Pivot row selection.
    var piv: u32 = i;
    var best: f32 = abs(A[i * 8u + i]);
    for (var r: u32 = i + 1u; r < 8u; r = r + 1u) {
      let v = abs(A[r * 8u + i]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-20) {
      return Solve8Res(false, vec4<f32>(0.0), vec4<f32>(0.0));
    }

    // Swap rows if needed.
    if (piv != i) {
      for (var c: u32 = 0u; c < 8u; c = c + 1u) {
        let tmp = A[i * 8u + c];
        A[i * 8u + c] = A[piv * 8u + c];
        A[piv * 8u + c] = tmp;
      }
      let tb = b[i];
      b[i] = b[piv];
      b[piv] = tb;
    }

    let inv_p = 1.0 / A[i * 8u + i];
    for (var r: u32 = i + 1u; r < 8u; r = r + 1u) {
      let f = A[r * 8u + i] * inv_p;
      // Eliminate column i.
      for (var c: u32 = i; c < 8u; c = c + 1u) {
        A[r * 8u + c] = A[r * 8u + c] - f * A[i * 8u + c];
      }
      b[r] = b[r] - f * b[i];
    }
  }

  // Back substitution.
  var x: array<f32, 8>;
  for (var ii: i32 = 7; ii >= 0; ii = ii - 1) {
    let i: u32 = u32(ii);
    var s = b[i];
    for (var j: u32 = i + 1u; j < 8u; j = j + 1u) {
      s = s - A[i * 8u + j] * x[j];
    }
    x[i] = s / A[i * 8u + i];
  }

  return Solve8Res(
    true,
    vec4<f32>(x[0], x[1], x[2], x[3]),
    vec4<f32>(x[4], x[5], x[6], x[7])
  );
}

@compute @workgroup_size(WG, 1, 1)
fn optimize_lm_grid(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let p = gid.x;
  let lane = lid.x;

  load_shared(p, lane);
  let n_rows = params.u.x;
  let opt_steps = params.u.z;

  let sum_w = params.f0.x;
  let lambda_scale = params.f0.y;
  let lambda_floor = params.f0.z;
  let step_clip = params.f0.w;

  // Lambda update factors (LM style).
  let lambda_up: f32 = 10.0;
  let lambda_down: f32 = 0.3;

  // Relative improvement threshold (avoid tiny accept/reject thrash).
  let rel_improve_eps: f32 = 1e-6;

  if (lane == 0u) {
    wg_done = 0u;
    wg_lambda = lambda_floor;
  }
  workgroupBarrier();

  for (var step: u32 = 0u; step < opt_steps; step = step + 1u) {
    if (wg_done != 0u) { break; }

    // --------------------------------------------------------------------------
    // Pass 1: compute SSE, g = J^T W r, H = J^T W J at current constants.
    var sum: f32 = 0.0;
    var g0: vec4<f32> = vec4<f32>(0.0);
    var g1: vec4<f32> = vec4<f32>(0.0);
    // Naga doesn't accept scalar matrix constructors like `mat4x4<f32>(0.0)`.
    var h00: mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
    );
    var h01: mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
    );
    var h11: mat4x4<f32> = mat4x4<f32>(
      vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
    );

    for (var row: u32 = lane; row < n_rows; row = row + WG) {
      let out = eval_postfix_value_and_grad(p, row);
      let r = out.v - y[row];
      let wi = w[row];

      sum = sum + wi * r * r;

      // g = J^T W r   (NOTE: gradient of SSE is 2*g)
      let fac = wi * r;
      g0 = g0 + fac * out.g0;
      g1 = g1 + fac * out.g1;

      // H = J^T W J
      h00 = h00 + wi * outer4(out.g0, out.g0);
      h01 = h01 + wi * outer4(out.g0, out.g1);
      h11 = h11 + wi * outer4(out.g1, out.g1);
    }

    sh_loss[lane] = sum;
    sh_g0[lane] = g0;
    sh_g1[lane] = g1;
    sh_h00[lane] = h00;
    sh_h01[lane] = h01;
    sh_h11[lane] = h11;
    workgroupBarrier();

    // Reduce SSE, g, H across lanes.
    var stride: u32 = WG / 2u;
    loop {
      if (stride == 0u) { break; }
      if (lane < stride) {
        sh_loss[lane] = sh_loss[lane] + sh_loss[lane + stride];
        sh_g0[lane] = sh_g0[lane] + sh_g0[lane + stride];
        sh_g1[lane] = sh_g1[lane] + sh_g1[lane + stride];
        sh_h00[lane] = sh_h00[lane] + sh_h00[lane + stride];
        sh_h01[lane] = sh_h01[lane] + sh_h01[lane + stride];
        sh_h11[lane] = sh_h11[lane] + sh_h11[lane + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }

    if (lane == 0u) {
      wg_base_sse = sh_loss[0];

      // Initialize lambda from the diagonal scale on the first step.
      if (step == 0u) {
        let h00s = sh_h00[0];
        let h11s = sh_h11[0];
        var diag_mean: f32 = 0.125 * (
          h00s[0][0] + h00s[1][1] + h00s[2][2] + h00s[3][3] +
          h11s[0][0] + h11s[1][1] + h11s[2][2] + h11s[3][3]
        );
        if (!(diag_mean > 0.0) || !is_finite(diag_mean)) {
          diag_mean = 1.0;
        }
        wg_lambda = max(lambda_floor, lambda_scale * diag_mean);
      }

      if (!is_finite(wg_base_sse)) {
        wg_done = 1u;
      }
    }
    workgroupBarrier();
    if (wg_done != 0u) { break; }

    // --------------------------------------------------------------------------
    // Pass 2: try up to LM_MAX_TRIES dampings sequentially.
    for (var attempt: u32 = 0u; attempt < LM_MAX_TRIES; attempt = attempt + 1u) {
      if (lane == 0u) {
        wg_accept = 0u;

        // Build damped normal equation (H + lambda I) dx = g
        let h00s = sh_h00[0];
        let h01s = sh_h01[0];
        let h11s = sh_h11[0];

        // Build damped normal equation (H + lambda I) dx = g as a row-major 8x8 matrix.
        var A: array<f32, 64>;
        A[0] = h00s[0][0]; A[1] = h00s[1][0]; A[2] = h00s[2][0]; A[3] = h00s[3][0]; A[4] = h01s[0][0]; A[5] = h01s[1][0]; A[6] = h01s[2][0]; A[7] = h01s[3][0];
        A[8] = h00s[0][1]; A[9] = h00s[1][1]; A[10] = h00s[2][1]; A[11] = h00s[3][1]; A[12] = h01s[0][1]; A[13] = h01s[1][1]; A[14] = h01s[2][1]; A[15] = h01s[3][1];
        A[16] = h00s[0][2]; A[17] = h00s[1][2]; A[18] = h00s[2][2]; A[19] = h00s[3][2]; A[20] = h01s[0][2]; A[21] = h01s[1][2]; A[22] = h01s[2][2]; A[23] = h01s[3][2];
        A[24] = h00s[0][3]; A[25] = h00s[1][3]; A[26] = h00s[2][3]; A[27] = h00s[3][3]; A[28] = h01s[0][3]; A[29] = h01s[1][3]; A[30] = h01s[2][3]; A[31] = h01s[3][3];
        A[32] = h01s[0][0]; A[33] = h01s[1][0]; A[34] = h01s[2][0]; A[35] = h01s[3][0]; A[36] = h11s[0][0]; A[37] = h11s[1][0]; A[38] = h11s[2][0]; A[39] = h11s[3][0];
        A[40] = h01s[0][1]; A[41] = h01s[1][1]; A[42] = h01s[2][1]; A[43] = h01s[3][1]; A[44] = h11s[0][1]; A[45] = h11s[1][1]; A[46] = h11s[2][1]; A[47] = h11s[3][1];
        A[48] = h01s[0][2]; A[49] = h01s[1][2]; A[50] = h01s[2][2]; A[51] = h01s[3][2]; A[52] = h11s[0][2]; A[53] = h11s[1][2]; A[54] = h11s[2][2]; A[55] = h11s[3][2];
        A[56] = h01s[0][3]; A[57] = h01s[1][3]; A[58] = h01s[2][3]; A[59] = h01s[3][3]; A[60] = h11s[0][3]; A[61] = h11s[1][3]; A[62] = h11s[2][3]; A[63] = h11s[3][3];

        // Add damping to diagonal.
        let lam = wg_lambda;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
          A[i * 8u + i] = A[i * 8u + i] + lam;
        }

        let g0s = sh_g0[0];
        let g1s = sh_g1[0];
        var b: array<f32, 8>;
        b[0] = g0s.x; b[1] = g0s.y; b[2] = g0s.z; b[3] = g0s.w;
        b[4] = g1s.x; b[5] = g1s.y; b[6] = g1s.z; b[7] = g1s.w;

        let sol = solve8(A, b);
        if (!sol.ok) {
          // If we can't solve, just stop.
          wg_done = 1u;
          wg_trial_c0 = wg_c0;
          wg_trial_c1 = wg_c1;
        } else {
          var dx0 = sol.dx0;
          var dx1 = sol.dx1;

          // Optional per-parameter clipping (helps avoid wild steps).
          if (step_clip > 0.0) {
            dx0 = clamp(dx0, vec4<f32>(-step_clip), vec4<f32>(step_clip));
            dx1 = clamp(dx1, vec4<f32>(-step_clip), vec4<f32>(step_clip));
          }

          wg_trial_c0 = wg_c0 - dx0;
          wg_trial_c1 = wg_c1 - dx1;
        }
      }
      workgroupBarrier();
      if (wg_done != 0u) { break; }

      // Evaluate SSE for the trial constants (full workgroup, no cand-splitting).
      var trial_sum: f32 = 0.0;
      for (var row: u32 = lane; row < n_rows; row = row + WG) {
        let pred = eval_postfix_value_with_consts(row, wg_trial_c0, wg_trial_c1);
        let r = pred - y[row];
        let wi = w[row];
        trial_sum = trial_sum + wi * r * r;
      }

      sh_loss[lane] = trial_sum;
      workgroupBarrier();

      // Reduce trial SSE.
      var stride2: u32 = WG / 2u;
      loop {
        if (stride2 == 0u) { break; }
        if (lane < stride2) {
          sh_loss[lane] = sh_loss[lane] + sh_loss[lane + stride2];
        }
        workgroupBarrier();
        stride2 = stride2 / 2u;
      }

      if (lane == 0u) {
        wg_trial_sse = sh_loss[0];

        // Accept if strictly improves.
        let thresh = wg_base_sse * (1.0 - rel_improve_eps);
        if (is_finite(wg_trial_sse) && (wg_trial_sse < thresh)) {
          wg_accept = 1u;
          wg_base_sse = wg_trial_sse;
          wg_c0 = wg_trial_c0;
          wg_c1 = wg_trial_c1;
          wg_lambda = max(lambda_floor, wg_lambda * lambda_down);
        } else {
          // Reject: increase damping and try again.
          wg_lambda = wg_lambda * lambda_up;
          if (wg_lambda < lambda_floor) { wg_lambda = lambda_floor; }
          if (attempt + 1u >= LM_MAX_TRIES) {
            // Give up this program: no improvement even with higher damping.
            wg_done = 1u;
          }
        }
      }
      workgroupBarrier();

      if (wg_accept != 0u) { break; }
      if (wg_done != 0u) { break; }
    }
    workgroupBarrier();
  }

  if (lane == 0u) {
    // Write back constants and final loss.
    store_consts(p);
    out_loss[p] = wg_base_sse / sum_w;
  }
}
