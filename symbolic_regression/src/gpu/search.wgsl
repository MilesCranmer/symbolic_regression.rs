// GPU-full symbolic regression search kernels.
//
// Key idea: keep the entire population resident on the GPU (programs + constants + losses),
// evolve it for many generations without CPU readback, and only read back the best program
// at the end (or periodically).
//
// Program encoding matches the existing packed postfix encoding used by kernels.wgsl:
//   instr = (payload << 2) | kind
// kinds: 0=var, 1=const, 2=op, 3=end
// payload:
//   var: feature index
//   const: constant index [0..MAX_CONSTS)
//   op: (arity<<8)|opcode
//
// NOTE: This is a GPU-friendly evolutionary strategy (tournament selection + mutation).
// Crossover and heavy simplification are intentionally omitted for performance.

const MAX_NODES: u32 = 32u;
const MAX_STACK: u32 = 64u;
const MAX_CONSTS: u32 = 8u;
const WG: u32 = 64u;
const NAN_F32: f32 = 0x7fc00000;

const KIND_VAR: u32 = 0u;
const KIND_CONST: u32 = 1u;
const KIND_OP: u32 = 2u;
const KIND_END: u32 = 3u;

// Opcodes (must match Rust mapping / existing kernels).
// Unary
const OP_NEG: u32 = 0u;
const OP_SIN: u32 = 1u;
const OP_COS: u32 = 2u;
const OP_EXP: u32 = 3u;
const OP_LOG: u32 = 4u;
const OP_SQRT: u32 = 5u;
const N_UNARY: u32 = 6u;
// Binary
const OP_ADD: u32 = 0u;
const OP_SUB: u32 = 1u;
const OP_MUL: u32 = 2u;
const OP_DIV: u32 = 3u;
const N_BINARY: u32 = 4u;

struct SearchParams {
  u0: vec4<u32>, // (n_rows, n_features, pop_total, pop_per_island)
  u1: vec4<u32>, // (gen, parity_in, tournament_k, opt_iters)
  u2: vec4<u32>, // (allowed_unary_mask, allowed_binary_mask, mutate_rate_ppm, regen_rate_ppm)
  u3: vec4<u32>, // (seed, reserved, reserved, reserved)
  f0: vec4<f32>, // (sum_w, adam_lr, beta1, beta2)
  f1: vec4<f32>, // (adam_eps, step_clip, const_sigma, reserved)
};

struct State {
  best_loss_bits: atomic<u32>,
  best_index: atomic<u32>, // global index into the packed buffers (0..2*pop_total)
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;

// Two generations in one buffer: gen0 at [0..pop_total), gen1 at [pop_total..2*pop_total)
@group(0) @binding(3) var<storage, read_write> programs: array<u32>;
@group(0) @binding(4) var<storage, read_write> consts: array<f32>;
@group(0) @binding(5) var<storage, read_write> losses: array<f32>;

@group(0) @binding(6) var<storage, read_write> state: State;
@group(0) @binding(7) var<uniform> params: SearchParams;

// --- Workgroup shared state ---
var<workgroup> wg_prog: array<u32, 32>;
var<workgroup> wg_c0: vec4<f32>;
var<workgroup> wg_c1: vec4<f32>;
var<workgroup> wg_reduce: array<f32, 64>;
var<workgroup> wg_reduce_g0: array<vec4<f32>, 64>;
var<workgroup> wg_reduce_g1: array<vec4<f32>, 64>;
var<workgroup> wg_parent_global: u32;

fn reduce_sum(lane: u32, v: f32) -> f32 {
  wg_reduce[lane] = v;
  workgroupBarrier();

  var stride: u32 = WG / 2u;
  loop {
    if (lane < stride) {
      wg_reduce[lane] = wg_reduce[lane] + wg_reduce[lane + stride];
    }
    workgroupBarrier();

    if (stride <= 1u) { break; }
    stride = stride / 2u;
  }
  return wg_reduce[0u];
}

fn reduce_sum_vec4(lane: u32, v: vec4<f32>, which: u32) -> vec4<f32> {
  if (which == 0u) {
    wg_reduce_g0[lane] = v;
  } else {
    wg_reduce_g1[lane] = v;
  }
  workgroupBarrier();

  var stride: u32 = WG / 2u;
  loop {
    if (lane < stride) {
      if (which == 0u) {
        wg_reduce_g0[lane] = wg_reduce_g0[lane] + wg_reduce_g0[lane + stride];
      } else {
        wg_reduce_g1[lane] = wg_reduce_g1[lane] + wg_reduce_g1[lane + stride];
      }
    }
    workgroupBarrier();

    if (stride <= 1u) { break; }
    stride = stride / 2u;
  }
  if (which == 0u) {
    return wg_reduce_g0[0u];
  }
  return wg_reduce_g1[0u];
}

fn const_value(i: u32) -> f32 {
  if (i < 4u) {
    return wg_c0[i];
  }
  return wg_c1[i - 4u];
}

fn set_const_value(i: u32, v: f32) {
  if (i < 4u) {
    wg_c0[i] = v;
  } else {
    wg_c1[i - 4u] = v;
  }
}

fn load_shared(p_global: u32, lane: u32) {
  // Load program tokens
  if (lane < MAX_NODES) {
    let off = p_global * MAX_NODES + lane;
    wg_prog[lane] = programs[off];
  }
  // Load constants (lane 0)
  if (lane == 0u) {
    let off = p_global * MAX_CONSTS;
    wg_c0 = vec4<f32>(consts[off + 0u], consts[off + 1u], consts[off + 2u], consts[off + 3u]);
    wg_c1 = vec4<f32>(consts[off + 4u], consts[off + 5u], consts[off + 6u], consts[off + 7u]);
  }
}

fn store_program(p_global: u32, lane: u32) {
  if (lane < MAX_NODES) {
    let off = p_global * MAX_NODES + lane;
    programs[off] = wg_prog[lane];
  }
}

fn store_consts(p_global: u32) {
  let off = p_global * MAX_CONSTS;
  consts[off + 0u] = wg_c0.x;
  consts[off + 1u] = wg_c0.y;
  consts[off + 2u] = wg_c0.z;
  consts[off + 3u] = wg_c0.w;
  consts[off + 4u] = wg_c1.x;
  consts[off + 5u] = wg_c1.y;
  consts[off + 6u] = wg_c1.z;
  consts[off + 7u] = wg_c1.w;
}

fn op_unary(op: u32, a: f32) -> f32 {
  switch op {
    case OP_NEG: { return -a; }
    case OP_SIN: { return sin(a); }
    case OP_COS: { return cos(a); }
    case OP_EXP: { return exp(a); }
    case OP_LOG: { return log(a); }
    case OP_SQRT: { return sqrt(a); }
    default: { return a; }
  }
}

fn op_binary(op: u32, a: f32, b: f32) -> f32 {
  switch op {
    case OP_ADD: { return a + b; }
    case OP_SUB: { return a - b; }
    case OP_MUL: { return a * b; }
    case OP_DIV: { return a / b; }
    default: { return a + b; }
  }
}

// Evaluate yhat for a single row.
fn eval_postfix_value(row: u32) -> f32 {
  let n_rows = params.u0.x;

  var stack: array<f32, 64>;
  var sp: u32 = 0u;

  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    let instr = wg_prog[i];
    let kind = instr & 3u;
    if (kind == KIND_END) {
      break;
    }
    let payload = instr >> 2u;

    if (kind == KIND_VAR) {
      if (sp >= MAX_STACK) {
        return NAN_F32;
      }
      let feat = payload;
      let idx = feat * n_rows + row;
      stack[sp] = x[idx];
      sp = sp + 1u;
    } else if (kind == KIND_CONST) {
      if (sp >= MAX_STACK) {
        return NAN_F32;
      }
      stack[sp] = const_value(payload);
      sp = sp + 1u;
    } else if (kind == KIND_OP) {
      let op_code = payload & 255u;
      let arity = payload >> 8u;
      if (arity == 1u) {
        if (sp < 1u) {
          return NAN_F32;
        }
        let a0 = stack[sp - 1u];
        stack[sp - 1u] = op_unary(op_code, a0);
      } else {
        if (sp < 2u) {
          return NAN_F32;
        }
        let b0 = stack[sp - 1u];
        let a0 = stack[sp - 2u];
        sp = sp - 1u;
        stack[sp - 1u] = op_binary(op_code, a0, b0);
      }
    }
  }
  if (sp != 1u) {
    return NAN_F32;
  }
  return stack[0u];
}

// Forward-mode gradient wrt each constant.
fn eval_postfix_value_and_grad(row: u32) -> array<f32, 9> {
  let n_rows = params.u0.x;

  var val_stack: array<f32, 64>;
  var grad_stack: array<vec4<f32>, 64>;
  var grad_stack2: array<vec4<f32>, 64>;
  var sp: u32 = 0u;

  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    let instr = wg_prog[i];
    let kind = instr & 3u;
    if (kind == KIND_END) {
      break;
    }
    let payload = instr >> 2u;

    if (kind == KIND_VAR) {
      if (sp >= MAX_STACK) {
        var out: array<f32, 9>;
        for (var j: u32 = 0u; j < 9u; j = j + 1u) { out[j] = NAN_F32; }
        return out;
      }
      let feat = payload;
      let idx = feat * n_rows + row;
      val_stack[sp] = x[idx];
      grad_stack[sp] = vec4<f32>(0.0);
      grad_stack2[sp] = vec4<f32>(0.0);
      sp = sp + 1u;
    } else if (kind == KIND_CONST) {
      if (sp >= MAX_STACK) {
        var out: array<f32, 9>;
        for (var j: u32 = 0u; j < 9u; j = j + 1u) { out[j] = NAN_F32; }
        return out;
      }
      val_stack[sp] = const_value(payload);
      // derivative is 1 for the referenced constant slot
      var g0 = vec4<f32>(0.0);
      var g1 = vec4<f32>(0.0);
      if (payload < 4u) {
        g0[payload] = 1.0;
      } else {
        g1[payload - 4u] = 1.0;
      }
      grad_stack[sp] = g0;
      grad_stack2[sp] = g1;
      sp = sp + 1u;
    } else if (kind == KIND_OP) {
      let op_code = payload & 255u;
      let arity = payload >> 8u;
      if (arity == 1u) {
        if (sp < 1u) {
          var out: array<f32, 9>;
          for (var j: u32 = 0u; j < 9u; j = j + 1u) { out[j] = NAN_F32; }
          return out;
        }
        let a = val_stack[sp - 1u];
        let g0a = grad_stack[sp - 1u];
        let g1a = grad_stack2[sp - 1u];
        // d/dx op(a)
        var d: f32 = 0.0;
        switch op_code {
          case OP_NEG: { d = -1.0; }
          case OP_SIN: { d = cos(a); }
          case OP_COS: { d = -sin(a); }
          case OP_EXP: { d = exp(a); }
          case OP_LOG: { d = 1.0 / a; }
          case OP_SQRT: { d = 0.5 / sqrt(a); }
          default: { d = 1.0; }
        }
        val_stack[sp - 1u] = op_unary(op_code, a);
        grad_stack[sp - 1u] = g0a * d;
        grad_stack2[sp - 1u] = g1a * d;
      } else {
        if (sp < 2u) {
          var out: array<f32, 9>;
          for (var j: u32 = 0u; j < 9u; j = j + 1u) { out[j] = NAN_F32; }
          return out;
        }
        let b = val_stack[sp - 1u];
        let a = val_stack[sp - 2u];
        let g0b = grad_stack[sp - 1u];
        let g0a = grad_stack[sp - 2u];
        let g1b = grad_stack2[sp - 1u];
        let g1a = grad_stack2[sp - 2u];
        // combine
        var out = 0.0;
        var g0 = vec4<f32>(0.0);
        var g1 = vec4<f32>(0.0);
        switch op_code {
          case OP_ADD: {
            out = a + b;
            g0 = g0a + g0b;
            g1 = g1a + g1b;
          }
          case OP_SUB: {
            out = a - b;
            g0 = g0a - g0b;
            g1 = g1a - g1b;
          }
          case OP_MUL: {
            out = a * b;
            g0 = g0a * b + g0b * a;
            g1 = g1a * b + g1b * a;
          }
          case OP_DIV: {
            out = a / b;
            let invb = 1.0 / b;
            let invb2 = invb * invb;
            g0 = g0a * invb + g0b * (-a * invb2);
            g1 = g1a * invb + g1b * (-a * invb2);
          }
          default: {
            out = a + b;
            g0 = g0a + g0b;
            g1 = g1a + g1b;
          }
        }
        sp = sp - 1u;
        val_stack[sp - 1u] = out;
        grad_stack[sp - 1u] = g0;
        grad_stack2[sp - 1u] = g1;
      }
    }
  }

  // Output: [value, grad0..grad7]
  if (sp != 1u) {
    var out: array<f32, 9>;
    for (var j: u32 = 0u; j < 9u; j = j + 1u) { out[j] = NAN_F32; }
    return out;
  }
  var out: array<f32, 9>;
  out[0] = val_stack[0u];
  let g0 = grad_stack[0u];
  let g1 = grad_stack2[0u];
  out[1] = g0.x;
  out[2] = g0.y;
  out[3] = g0.z;
  out[4] = g0.w;
  out[5] = g1.x;
  out[6] = g1.y;
  out[7] = g1.z;
  out[8] = g1.w;
  return out;
}

// --- RNG helpers ---
fn hash32(x: u32) -> u32 {
  var v = x;
  v = v ^ (v >> 16u);
  v = v * 0x7feb352du;
  v = v ^ (v >> 15u);
  v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

fn rng_init(idx: u32) -> u32 {
  let seed = params.u3.x;
  let gen = params.u1.x;
  // Mix seed, generation, and idx.
  return hash32(seed ^ (gen * 0x9e3779b9u) ^ (idx * 0x85ebca6bu));
}

fn rng_next(state: ptr<function, u32>) -> u32 {
  var x = *state;
  x = x ^ (x << 13u);
  x = x ^ (x >> 17u);
  x = x ^ (x << 5u);
  *state = x;
  return x;
}

fn rng_f32(state: ptr<function, u32>) -> f32 {
  let u = rng_next(state);
  // Convert to [0,1)
  return f32(u) * (1.0 / 4294967296.0);
}

fn rng_range(state: ptr<function, u32>, n: u32) -> u32 {
  // Simple modulo. Good enough for our purposes.
  return rng_next(state) % n;
}

fn rng_ppm(state: ptr<function, u32>) -> u32 {
  return rng_next(state) % 1000000u;
}

fn rng_norm4(state: ptr<function, u32>) -> f32 {
  // Approx N(0,1) using sum of uniforms.
  let s = rng_f32(state) + rng_f32(state) + rng_f32(state) + rng_f32(state);
  // mean 2, var 4/12=1/3, so std=sqrt(1/3)=0.577...
  // scale to std~1
  return (s - 2.0) * 1.7320508;
}

// --- Program helpers ---
fn make_var(feat: u32) -> u32 {
  return (feat << 2u) | KIND_VAR;
}

fn make_const(ci: u32) -> u32 {
  return (ci << 2u) | KIND_CONST;
}

fn make_op(arity: u32, opcode: u32) -> u32 {
  let payload = (arity << 8u) | opcode;
  return (payload << 2u) | KIND_OP;
}

fn allowed_pick(mask: u32, nbits: u32, state: ptr<function, u32>) -> u32 {
  if (mask == 0u) {
    return 0u;
  }
  // Rejection sample until hit a set bit.
  for (var i: u32 = 0u; i < 32u; i = i + 1u) {
    let k = rng_range(state, nbits);
    if (((mask >> k) & 1u) != 0u) {
      return k;
    }
  }
  // Fallback: first set bit.
  for (var k: u32 = 0u; k < 32u; k = k + 1u) {
    if (((mask >> k) & 1u) != 0u) {
      return k;
    }
  }
  return 0u;
}

fn random_consts(state: ptr<function, u32>) {
  // Initialize constants in [-1, 1].
  for (var i: u32 = 0u; i < MAX_CONSTS; i = i + 1u) {
    let v = (rng_f32(state) * 2.0) - 1.0;
    set_const_value(i, v);
  }
}

fn random_program(state: ptr<function, u32>) {
  let n_features = params.u0.y;
  let unary_mask = params.u2.x;
  let binary_mask = params.u2.y;

  var depth: i32 = 0;
  var wrote_end: bool = false;

  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    if (wrote_end) {
      wg_prog[i] = KIND_END;
      continue;
    }

    // Optional early stop if we already have a single value.
    if (depth == 1 && i > 3u) {
      if (rng_f32(state) < 0.08) {
        wg_prog[i] = KIND_END;
        wrote_end = true;
        continue;
      }
    }

    let remaining = i32(MAX_NODES - i);
    let min_needed = max(depth - 1, 0); // binary ops needed to reduce to 1

    if (depth <= 0) {
      // Must push
      if (rng_f32(state) < 0.7) {
        let f = rng_range(state, max(n_features, 1u));
        wg_prog[i] = make_var(f);
      } else {
        let c = rng_range(state, MAX_CONSTS);
        wg_prog[i] = make_const(c);
      }
      depth = depth + 1;
      continue;
    }

    if (remaining <= min_needed) {
      // Must apply binary ops until depth reduces.
      let op = allowed_pick(binary_mask, N_BINARY, state);
      wg_prog[i] = make_op(2u, op);
      depth = depth - 1;
      continue;
    }

    // Choose an action.
    let r = rng_f32(state);
    if (depth >= 2 && r < 0.40) {
      let op = allowed_pick(binary_mask, N_BINARY, state);
      wg_prog[i] = make_op(2u, op);
      depth = depth - 1;
    } else if (r < 0.55) {
      let op = allowed_pick(unary_mask, N_UNARY, state);
      wg_prog[i] = make_op(1u, op);
      // depth unchanged
    } else {
      if (rng_f32(state) < 0.7) {
        let f = rng_range(state, max(n_features, 1u));
        wg_prog[i] = make_var(f);
      } else {
        let c = rng_range(state, MAX_CONSTS);
        wg_prog[i] = make_const(c);
      }
      depth = depth + 1;
    }
  }

  // Ensure an END token exists.
  // If none was written, force last to END.
  if (!wrote_end) {
    wg_prog[MAX_NODES - 1u] = KIND_END;
  }
}

fn mutate_in_place(state: ptr<function, u32>) {
  let n_features = params.u0.y;
  let unary_mask = params.u2.x;
  let binary_mask = params.u2.y;
  let sigma = params.f1.z;

  // Find program length (first END).
  var len: u32 = 0u;
  for (var i: u32 = 0u; i < MAX_NODES; i = i + 1u) {
    if ((wg_prog[i] & 3u) == KIND_END) {
      len = i;
      break;
    }
    len = i + 1u;
  }
  if (len == 0u) {
    len = 1u;
  }

  // Mutate 1-3 instructions.
  let nm = 1u + (rng_next(state) % 3u);
  for (var m: u32 = 0u; m < nm; m = m + 1u) {
    let idx = rng_range(state, len);
    let instr = wg_prog[idx];
    let kind = instr & 3u;
    let payload = instr >> 2u;
    if (kind == KIND_VAR) {
      let f = rng_range(state, max(n_features, 1u));
      wg_prog[idx] = make_var(f);
    } else if (kind == KIND_CONST) {
      // Either swap which const is referenced, or just leave token alone.
      if (rng_f32(state) < 0.35) {
        let c = rng_range(state, MAX_CONSTS);
        wg_prog[idx] = make_const(c);
      }
    } else if (kind == KIND_OP) {
      let op_code = payload & 255u;
      let arity = payload >> 8u;
      if (arity == 1u) {
        let new_op = allowed_pick(unary_mask, N_UNARY, state);
        wg_prog[idx] = make_op(1u, new_op);
      } else {
        let new_op = allowed_pick(binary_mask, N_BINARY, state);
        wg_prog[idx] = make_op(2u, new_op);
      }
      _ = op_code;
    }
  }

  // Mutate constants with Gaussian-ish noise.
  for (var i: u32 = 0u; i < MAX_CONSTS; i = i + 1u) {
    if (rng_f32(state) < 0.40) {
      let v = const_value(i);
      let dv = rng_norm4(state) * sigma;
      set_const_value(i, v + dv);
    }
  }
}

fn update_best(p_global: u32, loss: f32) {
  // Loss is nonnegative; float->u32 bit order works for atomic min.
  var safe_loss = loss;
  // NaN check
  if (!(safe_loss == safe_loss)) {
    safe_loss = 3.402823e38;
  }
  if (safe_loss < 0.0) {
    safe_loss = 0.0;
  }
  let bits = bitcast<u32>(safe_loss);

  loop {
    let old = atomicLoad(&state.best_loss_bits);
    if (bits >= old) {
      break;
    }
    let res = atomicCompareExchangeWeak(&state.best_loss_bits, old, bits);
    if (res.exchanged) {
      atomicStore(&state.best_index, p_global);
      break;
    }
  }
}

fn compute_mse(lane: u32) -> f32 {
  let n_rows = params.u0.x;

  var sum: f32 = 0.0;
  for (var row: u32 = lane; row < n_rows; row = row + WG) {
    let yhat = eval_postfix_value(row);
    let r = yhat - y[row];
    sum = sum + w[row] * r * r;
  }
  let total = reduce_sum(lane, sum);
  if (lane == 0u) {
    let mse = total / params.f0.x;
    return mse;
  }
  return 0.0;
}

fn optimize_adam(lane: u32, iters: u32) -> f32 {
  let n_rows = params.u0.x;
  let sum_w = params.f0.x;
  let lr = params.f0.y;
  let b1 = params.f0.z;
  let b2 = params.f0.w;
  let eps = params.f1.x;
  let clip = params.f1.y;

  var m0 = vec4<f32>(0.0);
  var m1 = vec4<f32>(0.0);
  var v0 = vec4<f32>(0.0);
  var v1 = vec4<f32>(0.0);
  var last_loss: f32 = 3.402823e38;

  for (var iter: u32 = 0u; iter < iters; iter = iter + 1u) {
    // Accumulate loss + grad across rows handled by this lane.
    var s_loss: f32 = 0.0;
    var s_g0 = vec4<f32>(0.0);
    var s_g1 = vec4<f32>(0.0);

    for (var row: u32 = lane; row < n_rows; row = row + WG) {
      let out = eval_postfix_value_and_grad(row);
      let yhat = out[0];
      let r = yhat - y[row];
      let ww = w[row];
      s_loss = s_loss + ww * r * r;

      // d/dc (0.5 * r^2) = r * dyhat/dc
      s_g0 = s_g0 + ww * r * vec4<f32>(out[1], out[2], out[3], out[4]);
      s_g1 = s_g1 + ww * r * vec4<f32>(out[5], out[6], out[7], out[8]);
    }

    let total_loss = reduce_sum(lane, s_loss);
    let g0 = reduce_sum_vec4(lane, s_g0, 0u);
    let g1 = reduce_sum_vec4(lane, s_g1, 1u);

    if (lane == 0u) {
      // Normalize.
      var loss_now = total_loss / sum_w;
      if (!(loss_now == loss_now)) {
        loss_now = 3.402823e38;
      }
      last_loss = loss_now;

      let grad0 = g0 / sum_w;
      let grad1 = g1 / sum_w;

      // Adam update
      m0 = b1 * m0 + (1.0 - b1) * grad0;
      m1 = b1 * m1 + (1.0 - b1) * grad1;
      v0 = b2 * v0 + (1.0 - b2) * (grad0 * grad0);
      v1 = b2 * v1 + (1.0 - b2) * (grad1 * grad1);

      // Bias correction (approx). We avoid pow() in shader by using iterative factors.
      // For small iters, this is fine.
      let t = f32(iter + 1u);
      let b1t = pow(b1, t);
      let b2t = pow(b2, t);
      let m0h = m0 / (1.0 - b1t);
      let m1h = m1 / (1.0 - b1t);
      let v0h = v0 / (1.0 - b2t);
      let v1h = v1 / (1.0 - b2t);

      // Update consts with clipping.
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        var step = lr * m0h[j] / (sqrt(v0h[j]) + eps);
        step = clamp(step, -clip, clip);
        wg_c0[j] = wg_c0[j] - step;
      }
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        var step = lr * m1h[j] / (sqrt(v1h[j]) + eps);
        step = clamp(step, -clip, clip);
        wg_c1[j] = wg_c1[j] - step;
      }
    }

    workgroupBarrier();
  }

  // Broadcast last_loss via wg_reduce[0].
  if (lane == 0u) {
    wg_reduce[0u] = last_loss;
  }
  workgroupBarrier();
  return wg_reduce[0u];
}

// Initialize a random population into the CURRENT parity (params.u1.y) and compute losses.
@compute @workgroup_size(WG, 1, 1)
fn init_and_eval(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  let idx = gid.x;
  let pop_total = params.u0.z;
  if (idx >= pop_total) { return; }

  // Global index in the packed buffers for this individual.
  let parity = params.u1.y;
  let p_global = parity * pop_total + idx;

  // RNG
  var rng = rng_init(p_global);

  // Make fresh random program + constants.
  if (lane == 0u) {
    random_consts(&rng);
    random_program(&rng);
  }
  workgroupBarrier();

  // Optional constant optimization.
  let iters = params.u1.w;
  var loss_val: f32 = 0.0;
  if (iters > 0u) {
    loss_val = optimize_adam(lane, iters);
  } else {
    loss_val = compute_mse(lane);
  }

  // Write out.
  if (lane == 0u) {
    losses[p_global] = loss_val;
    store_consts(p_global);
    update_best(p_global, loss_val);
  }
  store_program(p_global, lane);
}

// Evolve one full generation:
//   read from parity_in (params.u1.y)
//   write to parity_out = 1 - parity_in
@compute @workgroup_size(WG, 1, 1)
fn evolve_generation(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  let idx = gid.x;
  let pop_total = params.u0.z;
  let pop_per_island = params.u0.w;
  if (idx >= pop_total) { return; }

  let parity_in = params.u1.y;
  let parity_out = 1u - parity_in;
  let base_in = parity_in * pop_total;
  let base_out = parity_out * pop_total;
  let p_out = base_out + idx;

  // RNG seeded by output index (so each child is deterministic across runs).
  var rng = rng_init(p_out);

  // Tournament selection within island.
  let island_base = (idx / pop_per_island) * pop_per_island;
  var best_local = island_base + rng_range(&rng, pop_per_island);
  var best_global = base_in + best_local;
  var best_loss = losses[best_global];

  let k = max(params.u1.z, 1u);
  for (var t: u32 = 1u; t < k; t = t + 1u) {
    let cand_local = island_base + rng_range(&rng, pop_per_island);
    let cand_global = base_in + cand_local;
    let l = losses[cand_global];
    // Handle NaNs as bad.
    let l_safe = select(3.402823e38, l, l == l);
    let best_safe = select(3.402823e38, best_loss, best_loss == best_loss);
    if (l_safe < best_safe) {
      best_local = cand_local;
      best_global = cand_global;
      best_loss = l;
    }
  }

  // Broadcast chosen parent to the workgroup.
  if (lane == 0u) {
    wg_parent_global = best_global;
  }
  workgroupBarrier();

  // Load parent into shared.
  load_shared(wg_parent_global, lane);
  workgroupBarrier();

  // Mutation / regeneration on lane 0.
  if (lane == 0u) {
    let regen_ppm = params.u2.w;
    let mutate_ppm = params.u2.z;

    if (rng_ppm(&rng) < regen_ppm) {
      // Full random restart
      random_consts(&rng);
      random_program(&rng);
    } else {
      // Copy parent already loaded; mutate with some probability.
      if (rng_ppm(&rng) < mutate_ppm) {
        mutate_in_place(&rng);
      }
    }
  }
  workgroupBarrier();

  // Optional constant optimization.
  let iters = params.u1.w;
  var loss_val: f32 = 0.0;
  if (iters > 0u) {
    loss_val = optimize_adam(lane, iters);
  } else {
    loss_val = compute_mse(lane);
  }

  // Write child to output.
  if (lane == 0u) {
    losses[p_out] = loss_val;
    store_consts(p_out);
    update_best(p_out, loss_val);
  }
  store_program(p_out, lane);
}
