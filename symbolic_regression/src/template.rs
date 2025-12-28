use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::expression::{ExpressionSpec, SRExpression};
use crate::operators::Operators;
use crate::options::Options;
use crate::pop_member::Evaluator;
use crate::{Dataset, constant_optimization};

#[derive(Clone, Copy)]
pub struct ValidVecView<'a, T> {
    pub x: &'a [T],
    pub valid: bool,
}

impl<'a, T> fmt::Debug for ValidVecView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidVecView")
            .field("len", &self.x.len())
            .field("valid", &self.valid)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct ValidVec<T> {
    pub x: Vec<T>,
    pub valid: bool,
}

impl<T> ValidVec<T> {
    pub fn view(&self) -> ValidVecView<'_, T> {
        ValidVecView {
            x: &self.x,
            valid: self.valid,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParamVector<T> {
    values: Vec<T>,
}

impl<T> ParamVector<T> {
    pub fn new(values: Vec<T>) -> Self {
        Self { values }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl<T> std::ops::Index<usize> for ParamVector<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[idx]
    }
}

pub trait TemplateContext<'a, T, Ops, const D: usize> {
    fn call(&mut self, name: &str, args: &[ValidVecView<'a, T>]) -> ValidVec<T>;
    fn param(&self, name: &str) -> Option<&ParamVector<T>>;
    fn n_rows(&self) -> usize;
}

pub type CombineFn<T, Ops, const D: usize> =
    dyn for<'a> Fn(&mut dyn TemplateContext<'a, T, Ops, D>, &[ValidVecView<'a, T>]) -> ValidVec<T> + Send + Sync;

pub struct TemplateStructure<T, Ops, const D: usize> {
    pub function_names: Vec<String>,
    pub function_arity: Vec<usize>,
    pub fn_index: HashMap<String, usize>,

    pub param_names: Vec<String>,
    pub param_len: Vec<usize>,
    pub param_index: HashMap<String, usize>,

    pub combine: Arc<CombineFn<T, Ops, D>>,
}

impl<T, Ops, const D: usize> fmt::Debug for TemplateStructure<T, Ops, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemplateStructure")
            .field("function_names", &self.function_names)
            .field("function_arity", &self.function_arity)
            .field("param_names", &self.param_names)
            .field("param_len", &self.param_len)
            .finish()
    }
}

impl<T, Ops, const D: usize> TemplateStructure<T, Ops, D> {
    pub fn new<F>(functions: Vec<(&str, usize)>, params: Vec<(&str, usize)>, combine: F) -> Self
    where
        F: for<'a> Fn(&mut dyn TemplateContext<'a, T, Ops, D>, &[ValidVecView<'a, T>]) -> ValidVec<T>
            + Send
            + Sync
            + 'static,
    {
        let mut function_names: Vec<String> = Vec::with_capacity(functions.len());
        let mut function_arity: Vec<usize> = Vec::with_capacity(functions.len());
        let mut fn_index: HashMap<String, usize> = HashMap::with_capacity(functions.len());
        for (i, (name, arity)) in functions.into_iter().enumerate() {
            function_names.push(name.to_string());
            function_arity.push(arity);
            fn_index.insert(name.to_string(), i);
        }

        let mut param_names: Vec<String> = Vec::with_capacity(params.len());
        let mut param_len: Vec<usize> = Vec::with_capacity(params.len());
        let mut param_index: HashMap<String, usize> = HashMap::with_capacity(params.len());
        for (i, (name, len)) in params.into_iter().enumerate() {
            param_names.push(name.to_string());
            param_len.push(len);
            param_index.insert(name.to_string(), i);
        }

        Self {
            function_names,
            function_arity,
            fn_index,
            param_names,
            param_len,
            param_index,
            combine: Arc::new(combine),
        }
    }

    pub fn n_functions(&self) -> usize {
        self.function_names.len()
    }

    pub fn n_params(&self) -> usize {
        self.param_names.len()
    }
}

#[derive(Clone)]
pub struct TemplatePlan<const D: usize> {
    pub subplans: Vec<dynamic_expressions::EvalPlan<D>>,
}

impl<const D: usize> fmt::Debug for TemplatePlan<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemplatePlan")
            .field("n_subplans", &self.subplans.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct TemplateExpression<T, Ops, const D: usize> {
    pub structure: Arc<TemplateStructure<T, Ops, D>>,
    pub trees: Vec<dynamic_expressions::PostfixExpr<T, Ops, D>>,
    pub params: Vec<ParamVector<T>>,
}

impl<T: Clone, Ops, const D: usize> Clone for TemplateExpression<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            structure: self.structure.clone(),
            trees: self.trees.clone(),
            params: self.params.clone(),
        }
    }
}

impl<T, Ops, const D: usize> fmt::Display for TemplateExpression<T, Ops, D>
where
    T: fmt::Display,
    dynamic_expressions::PostfixExpr<T, Ops, D>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TemplateExpression(")?;
        for (i, name) in self.structure.function_names.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let t = self.trees.get(i);
            match t {
                Some(t) => write!(f, "{name}={t}")?,
                None => write!(f, "{name}=<missing>")?,
            }
        }
        for (i, name) in self.structure.param_names.iter().enumerate() {
            write!(f, ", {name}=[")?;
            if let Some(p) = self.params.get(i) {
                for (j, v) in p.as_slice().iter().enumerate() {
                    if j > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{v}")?;
                }
            }
            write!(f, "]")?;
        }
        write!(f, ")")
    }
}

impl<T, Ops, const D: usize> TemplateExpression<T, Ops, D> {
    fn n_params_total(&self) -> usize {
        self.params.iter().map(|p| p.len()).sum()
    }

    fn n_const_total(&self) -> usize {
        self.trees.iter().map(|t| t.consts.len()).sum()
    }
}

struct EvalTemplateContext<'s, T, Ops, const D: usize> {
    structure: &'s TemplateStructure<T, Ops, D>,
    trees: &'s [dynamic_expressions::PostfixExpr<T, Ops, D>],
    plans: &'s [dynamic_expressions::EvalPlan<D>],
    params: &'s [ParamVector<T>],
    scratch: &'s mut ndarray::Array2<T>,
    eval_opts: &'s dynamic_expressions::EvalOptions,
    n_rows: usize,
}

impl<'a, 's, T, Ops, const D: usize> TemplateContext<'a, T, Ops, D> for EvalTemplateContext<'s, T, Ops, D>
where
    T: Float,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    fn call(&mut self, name: &str, args: &[ValidVecView<'a, T>]) -> ValidVec<T> {
        let idx = self
            .structure
            .fn_index
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("Unknown template function name: {name}"));
        let expected = self.structure.function_arity[idx];
        if args.len() != expected {
            panic!("Template call {name} got {} args, expected {expected}", args.len());
        }

        if args.iter().any(|a| !a.valid) {
            return ValidVec {
                x: vec![T::nan(); self.n_rows],
                valid: false,
            };
        }

        let mut out = vec![T::zero(); self.n_rows];
        let mut slices: Vec<&[T]> = Vec::with_capacity(args.len());
        for a in args {
            slices.push(a.x);
        }

        let ok = dynamic_expressions::eval_plan_slices_into::<T, Ops, D>(
            &mut out,
            &self.plans[idx],
            &self.trees[idx],
            &slices,
            self.scratch,
            self.eval_opts,
        );
        ValidVec { x: out, valid: ok }
    }

    fn param(&self, name: &str) -> Option<&ParamVector<T>> {
        let idx = self.structure.param_index.get(name).copied()?;
        self.params.get(idx)
    }

    fn n_rows(&self) -> usize {
        self.n_rows
    }
}

pub struct TemplateSpec<T, Ops, const D: usize> {
    pub structure: Arc<TemplateStructure<T, Ops, D>>,
}

impl<T, Ops, const D: usize> Clone for TemplateSpec<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            structure: self.structure.clone(),
        }
    }
}

impl<T, Ops, const D: usize> fmt::Debug for TemplateSpec<T, Ops, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemplateSpec")
            .field("structure", &self.structure)
            .finish()
    }
}

impl<T, Ops, const D: usize> TemplateSpec<T, Ops, D> {
    pub fn new(structure: Arc<TemplateStructure<T, Ops, D>>) -> Self {
        Self { structure }
    }
}

impl<T, Ops, const D: usize> ExpressionSpec<T, Ops, D> for TemplateSpec<T, Ops, D>
where
    T: Float + Clone + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    type Expr = TemplateExpression<T, Ops, D>;

    fn random_expr(
        &self,
        rng: &mut Rng,
        operators: &Operators<D>,
        _dataset_n_features: usize,
        target_size: usize,
        _options: &Options<T, D>,
    ) -> Self::Expr {
        let k = self.structure.n_functions().max(1);
        let total = target_size.max(k);
        let mut sizes = vec![1usize; k];
        for _ in 0..(total - k) {
            sizes[rng.usize(0..k)] += 1;
        }

        let mut trees = Vec::with_capacity(k);
        for (i, &arity) in self.structure.function_arity.iter().enumerate() {
            let sz = sizes.get(i).copied().unwrap_or(1);
            trees.push(crate::mutation_functions::random_expr(rng, operators, arity, sz));
        }

        let mut params = Vec::with_capacity(self.structure.n_params());
        for &len in &self.structure.param_len {
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let v = crate::random::standard_normal(rng);
                values.push(T::from(v).unwrap_or_else(T::zero));
            }
            params.push(ParamVector::new(values));
        }

        TemplateExpression {
            structure: self.structure.clone(),
            trees,
            params,
        }
    }
}

impl<T, Ops, const D: usize> SRExpression<T, Ops, D> for TemplateExpression<T, Ops, D>
where
    T: Float + Clone + Send + Sync,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    type Plan = TemplatePlan<D>;
    type MutationContext = usize;

    fn build_plan(&self, _dataset_n_features: usize) -> Self::Plan {
        assert_eq!(
            self.trees.len(),
            self.structure.n_functions(),
            "TemplateExpression trees len mismatch"
        );
        let mut subplans = Vec::with_capacity(self.trees.len());
        for (i, t) in self.trees.iter().enumerate() {
            let arity = self.structure.function_arity[i];
            subplans.push(dynamic_expressions::compile_plan(&t.nodes, arity, t.consts.len()));
        }
        TemplatePlan { subplans }
    }

    fn eval_with_plan(
        &self,
        plan: &Self::Plan,
        x: ndarray::ArrayView2<'_, T>,
        evaluator: &mut Evaluator<T, D>,
        eval_options: &dynamic_expressions::EvalOptions,
    ) -> bool {
        assert!(x.is_standard_layout(), "X columns must be contiguous");
        let x_data = x.as_slice().expect("X columns must be contiguous in memory");
        let n_rows = x.ncols();
        let n_features = x.nrows();
        evaluator.ensure_n_rows(n_rows);

        let mut inputs: Vec<ValidVecView<'_, T>> = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let start = f * n_rows;
            inputs.push(ValidVecView {
                x: &x_data[start..start + n_rows],
                valid: true,
            });
        }

        let mut ctx = EvalTemplateContext {
            structure: &self.structure,
            trees: &self.trees,
            plans: &plan.subplans,
            params: &self.params,
            scratch: &mut evaluator.scratch,
            eval_opts: eval_options,
            n_rows,
        };

        let out = (self.structure.combine)(&mut ctx, &inputs);
        if !out.valid || out.x.len() != n_rows {
            return false;
        }
        evaluator.yhat.copy_from_slice(&out.x);
        true
    }

    fn complexity(&self, options: &Options<T, D>) -> usize {
        self.trees
            .iter()
            .map(|t| crate::complexity::compute_complexity(&t.nodes, options))
            .sum()
    }

    fn check_constraints(&self, options: &Options<T, D>, curmaxsize: usize) -> bool {
        let total = self.complexity(options);
        if total > curmaxsize {
            return false;
        }
        for t in &self.trees {
            if !crate::check_constraints::check_constraints(t, options, curmaxsize) {
                return false;
            }
        }
        true
    }

    fn compress_constants(&mut self) {
        for t in &mut self.trees {
            dynamic_expressions::compress_constants(t);
        }
    }

    fn simplify_in_place(&mut self, eval_opts: &dynamic_expressions::EvalOptions) -> bool {
        let mut any = false;
        for t in &mut self.trees {
            any |= dynamic_expressions::simplify_in_place(t, eval_opts);
        }
        any
    }

    fn get_contents_for_mutation(
        &self,
        rng: &mut Rng,
    ) -> (dynamic_expressions::PostfixExpr<T, Ops, D>, Self::MutationContext) {
        let idx = rng.usize(0..self.trees.len());
        (self.trees[idx].clone(), idx)
    }

    fn with_contents_for_mutation(
        &self,
        mutated: dynamic_expressions::PostfixExpr<T, Ops, D>,
        ctx: Self::MutationContext,
    ) -> Self {
        let mut out = self.clone();
        out.trees[ctx] = mutated;
        out
    }

    fn nfeatures_for_mutation(&self, ctx: Self::MutationContext, _dataset_n_features: usize) -> usize {
        self.structure.function_arity[ctx]
    }

    fn feature_mutation_possible(&self, _dataset_n_features: usize) -> bool {
        self.structure.function_arity.iter().any(|&a| a > 1)
    }

    fn is_leaf(&self) -> bool {
        self.trees.iter().all(|t| {
            t.nodes.iter().all(|n| {
                matches!(
                    n,
                    dynamic_expressions::PNode::Var { .. } | dynamic_expressions::PNode::Const { .. }
                )
            })
        })
    }

    fn has_binary_op(&self) -> bool {
        self.trees.iter().any(|t| {
            t.nodes
                .iter()
                .any(|n| matches!(n, dynamic_expressions::PNode::Op { arity: 2, .. }))
        })
    }

    fn count_constant_nodes(&self) -> usize {
        self.trees
            .iter()
            .map(|t| dynamic_expressions::count_constant_nodes(&t.nodes))
            .sum()
    }

    fn count_scalar_constants(&self) -> usize {
        self.n_const_total() + self.n_params_total()
    }

    fn get_scalar_constants_flat(&self, out: &mut Vec<T>) {
        out.clear();
        for t in &self.trees {
            out.extend_from_slice(&t.consts);
        }
        for p in &self.params {
            out.extend_from_slice(p.as_slice());
        }
    }

    fn set_scalar_constants_flat(&mut self, values: &[T]) {
        let mut i = 0usize;
        for t in &mut self.trees {
            let n = t.consts.len();
            t.consts.clone_from_slice(&values[i..i + n]);
            i += n;
        }
        for p in &mut self.params {
            let n = p.values.len();
            p.values.clone_from_slice(&values[i..i + n]);
            i += n;
        }
        assert_eq!(i, values.len(), "scalar constants length mismatch");
    }

    fn loss_and_grad(
        &mut self,
        plan: &Self::Plan,
        dataset: &Dataset<T>,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
        grad_ctx: &mut dynamic_expressions::GradContext<T, D>,
        eval_opts: &dynamic_expressions::EvalOptions,
        grad_out: &mut [f64],
    ) -> Option<f64>
    where
        T: FromPrimitive + ToPrimitive + core::ops::AddAssign,
    {
        constant_optimization::finite_diff_loss_and_grad::<T, Ops, D, Self>(
            self, plan, dataset, options, evaluator, grad_ctx, eval_opts, grad_out,
        )
    }

    fn mutate_constant(&mut self, rng: &mut Rng, temperature: f64, options: &Options<T, D>) -> bool {
        let n_params = self.n_params_total();
        let n_const_nodes = self.count_constant_nodes();
        if n_params == 0 && n_const_nodes == 0 {
            return false;
        }

        let mutate_params = if n_params == 0 {
            false
        } else if n_const_nodes == 0 {
            true
        } else {
            rng.bool()
        };

        if mutate_params {
            let mut idx = rng.usize(0..n_params);
            for p in &mut self.params {
                if idx < p.values.len() {
                    mutate_scalar(rng, &mut p.values[idx], temperature, options);
                    return true;
                }
                idx -= p.values.len();
            }
            return false;
        }

        // Mutate a constant node within a subexpression (weighted by const-node count).
        let mut total = 0usize;
        let mut weights: Vec<usize> = Vec::with_capacity(self.trees.len());
        for t in &self.trees {
            let c = dynamic_expressions::count_constant_nodes(&t.nodes);
            weights.push(c);
            total += c;
        }
        if total == 0 {
            return false;
        }
        let mut pick = rng.usize(0..total);
        for (i, &w) in weights.iter().enumerate() {
            if pick < w {
                return crate::mutation_functions::mutate_constant_in_place(
                    rng,
                    &mut self.trees[i],
                    temperature,
                    options,
                );
            }
            pick -= w;
        }
        false
    }

    fn randomize(
        &self,
        rng: &mut Rng,
        operators: &Operators<D>,
        _dataset_n_features: usize,
        target_size: usize,
        options: &Options<T, D>,
    ) -> Self {
        let k = self.structure.n_functions().max(1);
        let total = target_size.max(k);
        let mut sizes = vec![1usize; k];
        for _ in 0..(total - k) {
            sizes[rng.usize(0..k)] += 1;
        }

        let mut out = self.clone();
        for (i, t) in out.trees.iter_mut().enumerate() {
            let arity = out.structure.function_arity[i];
            let sz = sizes.get(i).copied().unwrap_or(1);
            *t = crate::mutation_functions::random_expr(rng, operators, arity, sz);
        }

        for p in &mut out.params {
            for v in &mut p.values {
                let r = crate::random::standard_normal(rng);
                *v = T::from(r).unwrap_or_else(T::zero);
            }
        }

        let _ = options;
        out
    }
}

fn mutate_scalar<T: Float, const D: usize>(rng: &mut Rng, v: &mut T, temperature: f64, options: &Options<T, D>) {
    // Follows SymbolicRegression.jl's `mutate_factor` (mirrors `mutate_constant_in_place`).
    let pf = options.perturbation_factor * temperature.max(0.0);
    let max_change = pf + 1.1;
    let exponent: f64 = rng.f64();
    let mut mul = max_change.powf(exponent);
    let make_bigger: bool = rng.bool();
    mul = if make_bigger { mul } else { 1.0 / mul };
    if rng.f64() > options.probability_negate_constant {
        mul = -mul;
    }
    *v = *v * T::from(mul).unwrap_or_else(T::one);
}
