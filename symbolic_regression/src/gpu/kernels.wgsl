const MAX_NODES: u32 = 32u;
const MAX_CONSTS: u32 = 8u;
const WG: u32 = 256u;

struct Params {
  n_rows: u32,
  n_features: u32,
  sum_w: f32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> programs: array<u32>;
@group(0) @binding(4) var<storage, read> consts: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_loss: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
// Only used by eval_mse_grad.
@group(0) @binding(7) var<storage, read_write> out_grad: array<f32>;

fn nan_f32() -> f32 {
  return f32(0x7fc00000);
}

fn op_unary_value(op: u32, a: f32) -> f32 {
  switch op {
    case 0u: { return -a; }      // Neg
    case 1u: { return sin(a); }  // Sin
    case 2u: { return cos(a); }  // Cos
    case 3u: { return exp(a); }  // Exp
    case 4u: { return log(a); }  // Log
    case 5u: { return sqrt(a); } // Sqrt
    default: { return nan_f32(); }
  }
}

// Returns f'(a) for the unary op (so derivative = f'(a) * da).
fn op_unary_deriv_mul(op: u32, a: f32, f_of_a: f32) -> f32 {
  switch op {
    case 0u: { return -1.0; }          // d(-a)/da
    case 1u: { return cos(a); }        // d(sin)/da
    case 2u: { return -sin(a); }       // d(cos)/da
    case 3u: { return f_of_a; }        // d(exp)/da = exp(a)
    case 4u: { return 1.0 / a; }       // d(log)/da
    case 5u: { return 0.5 / f_of_a; }  // d(sqrt)/da = 1/(2*sqrt(a))
    default: { return nan_f32(); }
  }
}

fn op_binary_value(op: u32, a: f32, b: f32) -> f32 {
  switch op {
    case 0u: { return a + b; } // Add
    case 1u: { return a - b; } // Sub
    case 2u: { return a * b; } // Mul
    case 3u: { return a / b; } // Div
    default: { return nan_f32(); }
  }
}

// Returns (df/da, df/db) for binary ops.
fn op_binary_partials(op: u32, a: f32, b: f32) -> vec2<f32> {
  switch op {
    case 0u: { return vec2<f32>(1.0, 1.0); }      // Add
    case 1u: { return vec2<f32>(1.0, -1.0); }     // Sub
    case 2u: { return vec2<f32>(b, a); }          // Mul
    case 3u: { return vec2<f32>(1.0 / b, -a / (b * b)); } // Div
    default: { return vec2<f32>(nan_f32(), nan_f32()); }
  }
}

fn eval_postfix(p: u32, row: u32) -> f32 {
  let prog_off: u32 = p * MAX_NODES;
  let const_off: u32 = p * MAX_CONSTS;
  let n_rows: u32 = params.n_rows;

  var stack: array<f32, MAX_NODES>;
  var sp: i32 = 0;

  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    let t: u32 = programs[prog_off + i];
    let kind: u32 = t & 3u;
    if (kind == 3u) { break; } // END

    if (kind == 0u) { // VAR
      let f: u32 = t >> 2u;
      let off: u32 = f * n_rows + row;
      stack[sp] = x[off];
      sp = sp + 1;
    } else if (kind == 1u) { // CONST
      let c: u32 = t >> 2u;
      stack[sp] = consts[const_off + c];
      sp = sp + 1;
    } else { // OP
      let arity: u32 = (t >> 2u) & 0xFFu;
      let op: u32 = t >> 10u;

      if (arity == 1u) {
        let a: f32 = stack[sp - 1];
        stack[sp - 1] = op_unary_value(op, a);
      } else if (arity == 2u) {
        let b: f32 = stack[sp - 1];
        let a: f32 = stack[sp - 2];
        sp = sp - 1;
        stack[sp - 1] = op_binary_value(op, a, b);
      } else {
        return nan_f32();
      }
    }
  }

  return stack[sp - 1];
}

var<workgroup> reduce_s: array<f32, WG>;

@compute @workgroup_size(256, 1, 1)
fn eval_mse(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let p: u32 = gid.x;
  let lane: u32 = lid.x;
  let n_rows: u32 = params.n_rows;

  var sum: f32 = 0.0;
  for (var row: u32 = lane; row < n_rows; row = row + WG) {
    let yhat = eval_postfix(p, row);
    let r = yhat - y[row];
    sum = sum + w[row] * (r * r);
  }

  reduce_s[lane] = sum;
  workgroupBarrier();

  var stride: u32 = WG / 2u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce_s[lane] = reduce_s[lane] + reduce_s[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lane == 0u) {
    out_loss[p] = reduce_s[0] / params.sum_w;
  }
}

struct DualOut {
  v: f32,
  g0: vec4<f32>, // const[0..3]
  g1: vec4<f32>, // const[4..7]
};

fn eval_postfix_dual(p: u32, row: u32) -> DualOut {
  let prog_off: u32 = p * MAX_NODES;
  let const_off: u32 = p * MAX_CONSTS;
  let n_rows: u32 = params.n_rows;

  var stack_v: array<f32, MAX_NODES>;
  var stack_g0: array<vec4<f32>, MAX_NODES>;
  var stack_g1: array<vec4<f32>, MAX_NODES>;
  var sp: i32 = 0;

  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    let t: u32 = programs[prog_off + i];
    let kind: u32 = t & 3u;
    if (kind == 3u) { break; } // END

    if (kind == 0u) { // VAR
      let f: u32 = t >> 2u;
      let off: u32 = f * n_rows + row;
      stack_v[sp] = x[off];
      stack_g0[sp] = vec4<f32>(0.0);
      stack_g1[sp] = vec4<f32>(0.0);
      sp = sp + 1;
    } else if (kind == 1u) { // CONST
      let c: u32 = t >> 2u;
      let cv: f32 = consts[const_off + c];
      stack_v[sp] = cv;

      var g0 = vec4<f32>(0.0);
      var g1 = vec4<f32>(0.0);
      if (c < 4u) {
        if (c == 0u) { g0.x = 1.0; }
        if (c == 1u) { g0.y = 1.0; }
        if (c == 2u) { g0.z = 1.0; }
        if (c == 3u) { g0.w = 1.0; }
      } else {
        let k: u32 = c - 4u;
        if (k == 0u) { g1.x = 1.0; }
        if (k == 1u) { g1.y = 1.0; }
        if (k == 2u) { g1.z = 1.0; }
        if (k == 3u) { g1.w = 1.0; }
      }

      stack_g0[sp] = g0;
      stack_g1[sp] = g1;
      sp = sp + 1;
    } else { // OP
      let arity: u32 = (t >> 2u) & 0xFFu;
      let op: u32 = t >> 10u;

      if (arity == 1u) {
        let a_v: f32 = stack_v[sp - 1];
        let a_g0: vec4<f32> = stack_g0[sp - 1];
        let a_g1: vec4<f32> = stack_g1[sp - 1];

        let v: f32 = op_unary_value(op, a_v);
        let mul: f32 = op_unary_deriv_mul(op, a_v, v);

        stack_v[sp - 1] = v;
        stack_g0[sp - 1] = a_g0 * mul;
        stack_g1[sp - 1] = a_g1 * mul;
      } else if (arity == 2u) {
        let b_v: f32 = stack_v[sp - 1];
        let b_g0: vec4<f32> = stack_g0[sp - 1];
        let b_g1: vec4<f32> = stack_g1[sp - 1];

        let a_v: f32 = stack_v[sp - 2];
        let a_g0: vec4<f32> = stack_g0[sp - 2];
        let a_g1: vec4<f32> = stack_g1[sp - 2];

        let v: f32 = op_binary_value(op, a_v, b_v);
        let muls: vec2<f32> = op_binary_partials(op, a_v, b_v);
        let g0: vec4<f32> = a_g0 * muls.x + b_g0 * muls.y;
        let g1: vec4<f32> = a_g1 * muls.x + b_g1 * muls.y;

        sp = sp - 1;
        stack_v[sp - 1] = v;
        stack_g0[sp - 1] = g0;
        stack_g1[sp - 1] = g1;
      } else {
        return DualOut(nan_f32(), vec4<f32>(nan_f32()), vec4<f32>(nan_f32()));
      }
    }
  }

  return DualOut(stack_v[sp - 1], stack_g0[sp - 1], stack_g1[sp - 1]);
}

var<workgroup> reduce_loss: array<f32, WG>;
var<workgroup> reduce_g0: array<vec4<f32>, WG>;
var<workgroup> reduce_g1: array<vec4<f32>, WG>;

@compute @workgroup_size(256, 1, 1)
fn eval_mse_grad(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let p: u32 = gid.x;
  let lane: u32 = lid.x;
  let n_rows: u32 = params.n_rows;

  var sum_loss: f32 = 0.0;
  var sum_g0: vec4<f32> = vec4<f32>(0.0);
  var sum_g1: vec4<f32> = vec4<f32>(0.0);

  for (var row: u32 = lane; row < n_rows; row = row + WG) {
    let out = eval_postfix_dual(p, row);
    let r = out.v - y[row];
    let wi = w[row];

    sum_loss = sum_loss + wi * (r * r);

    let fac = 2.0 * wi * r;
    sum_g0 = sum_g0 + fac * out.g0;
    sum_g1 = sum_g1 + fac * out.g1;
  }

  reduce_loss[lane] = sum_loss;
  reduce_g0[lane] = sum_g0;
  reduce_g1[lane] = sum_g1;
  workgroupBarrier();

  var stride: u32 = WG / 2u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce_loss[lane] = reduce_loss[lane] + reduce_loss[lane + stride];
      reduce_g0[lane] = reduce_g0[lane] + reduce_g0[lane + stride];
      reduce_g1[lane] = reduce_g1[lane] + reduce_g1[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lane == 0u) {
    if (params.sum_w == 0.0) {
      out_loss[p] = 0.0;
      for (var i: u32 = 0u; i < MAX_CONSTS; i = i + 1u) {
        out_grad[p * MAX_CONSTS + i] = 0.0;
      }
      return;
    }

    let inv = 1.0 / params.sum_w;
    out_loss[p] = reduce_loss[0] * inv;

    let g0 = reduce_g0[0] * inv;
    let g1 = reduce_g1[0] * inv;

    let base: u32 = p * MAX_CONSTS;
    out_grad[base + 0u] = g0.x;
    out_grad[base + 1u] = g0.y;
    out_grad[base + 2u] = g0.z;
    out_grad[base + 3u] = g0.w;
    out_grad[base + 4u] = g1.x;
    out_grad[base + 5u] = g1.y;
    out_grad[base + 6u] = g1.z;
    out_grad[base + 7u] = g1.w;
  }
}
