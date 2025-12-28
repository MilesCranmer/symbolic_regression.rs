use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use ndarray::{Array2, ArrayView2};
use num_traits::Float;

use crate::expression::PostfixExpr;
use crate::interfaces::{Evaluatable, HasTrees, ScalarConstants};
use crate::traits::OperatorSet;
use crate::{compile, evaluate, strings};

#[derive(Copy, Clone, Debug)]
enum TemplateColor {
    Magenta,
    Green,
    Red,
    Blue,
    Yellow,
    Cyan,
}

fn template_color_for(i: usize) -> TemplateColor {
    match i % 6 {
        0 => TemplateColor::Magenta,
        1 => TemplateColor::Green,
        2 => TemplateColor::Red,
        3 => TemplateColor::Blue,
        4 => TemplateColor::Yellow,
        _ => TemplateColor::Cyan,
    }
}

fn ansi_wrap(s: &str, color: TemplateColor) -> String {
    let code = match color {
        TemplateColor::Magenta => "35",
        TemplateColor::Green => "32",
        TemplateColor::Red => "31",
        TemplateColor::Blue => "34",
        TemplateColor::Yellow => "33",
        TemplateColor::Cyan => "36",
    };
    format!("\x1b[{code}m{s}\x1b[0m")
}

fn pipe_prefix(i: usize, n: usize) -> &'static str {
    if n <= 1 {
        return "";
    }
    if i == 0 {
        "╭ "
    } else if i + 1 == n {
        "╰ "
    } else {
        "├ "
    }
}

fn format_param_vec<T: fmt::Display>(values: &[T], max_elems: usize) -> String {
    let max_elems = max_elems.max(1);
    if values.is_empty() {
        return "[]".to_string();
    }

    let mut parts: Vec<String> = Vec::new();
    if values.len() <= max_elems {
        parts.extend(values.iter().map(|v| v.to_string()));
    } else if max_elems <= 2 {
        parts.push(values[0].to_string());
        parts.push("...".to_string());
        parts.push(values[values.len() - 1].to_string());
    } else {
        let head = max_elems - 2;
        parts.extend(values.iter().take(head).map(|v| v.to_string()));
        parts.push("...".to_string());
        parts.push(values[values.len() - 1].to_string());
    }

    format!("[{}]", parts.join(", "))
}

pub fn string_template_pretty<T, Ops, const D: usize>(expr: &TemplateExpression<T, Ops, D>) -> String
where
    T: fmt::Display,
    Ops: OperatorSet,
{
    let mut out = String::new();

    let max_param_elems = 5;
    let n_lines = expr.structure.n_functions() + expr.structure.n_params();
    out.push_str("TemplateExpression");
    if n_lines == 0 {
        return out;
    }

    for (i, name) in expr.structure.function_names.iter().enumerate() {
        let prefix = pipe_prefix(i, n_lines);
        let color = template_color_for(i);
        out.push('\n');
        out.push_str(prefix);
        out.push_str(name);
        out.push_str(" = ");
        let tree = expr.trees.get(i);
        match tree {
            None => out.push_str("<missing>"),
            Some(tree) => {
                let arity = expr.structure.function_arity.get(i).copied().unwrap_or(0);
                let var_names: Vec<String> = (1..=arity).map(|k| format!("#{k}")).collect();
                let s = strings::string_tree(
                    tree,
                    strings::StringTreeOptions {
                        variable_names: Some(&var_names),
                        pretty: true,
                    },
                );
                out.push_str(&ansi_wrap(&s, color));
            }
        }
    }

    for (j, name) in expr.structure.param_names.iter().enumerate() {
        let i = expr.structure.n_functions() + j;
        let prefix = pipe_prefix(i, n_lines);
        let color = template_color_for(i);
        out.push('\n');
        out.push_str(prefix);
        out.push_str(name);
        out.push_str(" = ");
        let p = expr.params.get(j);
        let s = format_param_vec(p.map(Vec::as_slice).unwrap_or(&[]), max_param_elems);
        out.push_str(&ansi_wrap(&s, color));
    }

    out
}

fn string_template_compact<T, Ops, const D: usize>(expr: &TemplateExpression<T, Ops, D>) -> String
where
    T: fmt::Display,
    Ops: OperatorSet,
{
    let max_param_elems = 5;
    let mut out = String::new();

    out.push_str("TemplateExpression(");
    for (i, name) in expr.structure.function_names.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let tree = expr.trees.get(i);
        match tree {
            None => {
                out.push_str(name);
                out.push_str("=<missing>");
            }
            Some(tree) => {
                let arity = expr.structure.function_arity.get(i).copied().unwrap_or(0);
                let var_names: Vec<String> = (1..=arity).map(|k| format!("#{k}")).collect();
                let s = strings::string_tree(
                    tree,
                    strings::StringTreeOptions {
                        variable_names: Some(&var_names),
                        pretty: false,
                    },
                );
                out.push_str(name);
                out.push('=');
                out.push_str(&s);
            }
        }
    }
    for (i, name) in expr.structure.param_names.iter().enumerate() {
        out.push_str(", ");
        out.push_str(name);
        out.push('=');
        let p = expr.params.get(i);
        out.push_str(&format_param_vec(p.map(Vec::as_slice).unwrap_or(&[]), max_param_elems));
    }
    out.push(')');

    out
}

pub trait TemplateContext<T, Ops, const D: usize> {
    fn call(&mut self, name: &str, args: &[&[T]]) -> Vec<T>;
    fn param(&self, name: &str) -> Option<&[T]>;
    fn n_rows(&self) -> usize;
}

pub type CombineFn<T, Ops, const D: usize> =
    dyn for<'a> Fn(&mut dyn TemplateContext<T, Ops, D>, &[&'a [T]]) -> Vec<T> + Send + Sync;

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
        F: for<'a> Fn(&mut dyn TemplateContext<T, Ops, D>, &[&'a [T]]) -> Vec<T> + Send + Sync + 'static,
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

    pub fn new_fixed_inputs<const N: usize, F>(
        functions: Vec<(&str, usize)>,
        params: Vec<(&str, usize)>,
        combine: F,
    ) -> Self
    where
        F: for<'a> Fn(&mut dyn TemplateContext<T, Ops, D>, [&'a [T]; N]) -> Vec<T> + Send + Sync + 'static,
    {
        Self::new(functions, params, move |ctx, inputs| {
            let inputs: &[&[T]; N] = inputs
                .try_into()
                .unwrap_or_else(|_| panic!("template combine expected {N} inputs, got {}", inputs.len()));
            combine(ctx, *inputs)
        })
    }

    pub fn n_functions(&self) -> usize {
        self.function_names.len()
    }

    pub fn n_params(&self) -> usize {
        self.param_names.len()
    }
}

#[derive(Debug)]
pub struct TemplateExpression<T, Ops, const D: usize> {
    pub structure: Arc<TemplateStructure<T, Ops, D>>,
    pub trees: Vec<PostfixExpr<T, Ops, D>>,
    pub params: Vec<Vec<T>>,
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
    Ops: OperatorSet,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&string_template_compact(self))
    }
}

struct EvalTemplateContext<'s, T, Ops, const D: usize> {
    structure: &'s TemplateStructure<T, Ops, D>,
    trees: &'s [PostfixExpr<T, Ops, D>],
    plans: &'s [compile::EvalPlan<D>],
    params: &'s [Vec<T>],
    scratch: &'s mut Array2<T>,
    eval_opts: &'s evaluate::EvalOptions,
    n_rows: usize,
}

impl<'s, T, Ops, const D: usize> TemplateContext<T, Ops, D> for EvalTemplateContext<'s, T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn call(&mut self, name: &str, args: &[&[T]]) -> Vec<T> {
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

        let mut out = vec![T::zero(); self.n_rows];

        let ok = evaluate::eval_plan_slices_into::<T, Ops, D>(
            &mut out,
            &self.plans[idx],
            &self.trees[idx],
            args,
            self.scratch,
            self.eval_opts,
        );
        if !ok {
            out.fill(T::nan());
        }
        out
    }

    fn param(&self, name: &str) -> Option<&[T]> {
        let idx = self.structure.param_index.get(name).copied()?;
        Some(self.params.get(idx)?.as_slice())
    }

    fn n_rows(&self) -> usize {
        self.n_rows
    }
}

impl<T, Ops, const D: usize> HasTrees<T, Ops, D> for TemplateExpression<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn n_trees(&self) -> usize {
        self.trees.len()
    }

    fn tree(&self, i: usize) -> &PostfixExpr<T, Ops, D> {
        &self.trees[i]
    }

    fn tree_mut(&mut self, i: usize) -> &mut PostfixExpr<T, Ops, D> {
        &mut self.trees[i]
    }

    fn tree_nfeatures(&self, i: usize, _dataset_nfeatures: usize) -> usize {
        self.structure.function_arity[i]
    }
}

impl<T, Ops, const D: usize> Evaluatable<T, Ops, D> for TemplateExpression<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn eval_with_plans(
        &self,
        plans: &[compile::EvalPlan<D>],
        x: ArrayView2<'_, T>,
        out: &mut [T],
        scratch: &mut Array2<T>,
        eval_options: &evaluate::EvalOptions,
    ) -> bool {
        assert_eq!(plans.len(), self.trees.len(), "TemplateExpression plan len mismatch");
        assert!(x.is_standard_layout(), "X columns must be contiguous");
        let x_data = x.as_slice().expect("X columns must be contiguous in memory");
        let n_rows = x.ncols();
        let n_features = x.nrows();

        assert_eq!(out.len(), n_rows, "output length mismatch");

        let mut inputs: Vec<&[T]> = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let start = f * n_rows;
            inputs.push(&x_data[start..start + n_rows]);
        }

        let mut ctx = EvalTemplateContext {
            structure: &self.structure,
            trees: &self.trees,
            plans,
            params: &self.params,
            scratch,
            eval_opts: eval_options,
            n_rows,
        };

        let tmp = (self.structure.combine)(&mut ctx, &inputs);
        if tmp.len() != n_rows {
            return false;
        }
        if eval_options.check_finite && tmp.iter().any(|v| !v.is_finite()) {
            return false;
        }
        out.copy_from_slice(&tmp);
        true
    }
}

impl<T, Ops, const D: usize> ScalarConstants<T, Ops, D> for TemplateExpression<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    fn n_scalars(&self) -> usize {
        self.trees.iter().map(|t| t.consts.len()).sum::<usize>() + self.params.iter().map(Vec::len).sum::<usize>()
    }

    fn pack_scalars(&self, out: &mut Vec<T>) {
        out.clear();
        out.reserve(self.n_scalars());
        for t in &self.trees {
            out.extend_from_slice(&t.consts);
        }
        for p in &self.params {
            out.extend_from_slice(p.as_slice());
        }
    }

    fn unpack_scalars(&mut self, scalars: &[T]) {
        assert_eq!(scalars.len(), self.n_scalars(), "scalar constants length mismatch");
        let mut i = 0usize;
        for t in &mut self.trees {
            let n = t.consts.len();
            t.consts.clone_from_slice(&scalars[i..i + n]);
            i += n;
        }
        for p in &mut self.params {
            let n = p.len();
            p.clone_from_slice(&scalars[i..i + n]);
            i += n;
        }
        debug_assert_eq!(i, scalars.len());
    }
}
