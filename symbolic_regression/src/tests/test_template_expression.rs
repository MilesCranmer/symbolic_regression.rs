use dynamic_expressions::HasOp;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin;
use ndarray::{Array1, Array2};
use num_traits::{Float, Zero};

use super::common::{D, T, TestOps};
use crate::operator_library::OperatorLibrary;
use crate::pop_member::{Evaluator, MemberId, PopMember};
use crate::template::{ParamVector, TemplateExpression, TemplateSpec, TemplateStructure, ValidVec};
use crate::{Dataset, Options, SRExpression};

#[test]
fn template_expression_evaluates_and_matches_target() {
    let n_rows = 16;
    let n_features = 3;

    let mut x_data = Vec::with_capacity(n_features * n_rows);
    let mut y_data = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let x0 = i as T;
        let x1 = (i as T) * 2.0;
        let x2 = 1.0 + (i as T) * 0.5;
        x_data.push(x0);
        x_data.push(x1);
        x_data.push(x2);

        // y = (x0 + x1) + a0 * x2, with a0 = 2.0
        y_data.push((x0 + x1) + 2.0 * x2);
    }

    // Our Dataset layout is (n_features, n_rows).
    let mut x = vec![0.0; n_features * n_rows];
    for r in 0..n_rows {
        for f in 0..n_features {
            x[f * n_rows + r] = x_data[r * n_features + f];
        }
    }

    let dataset = Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y_data),
    );

    let options = Options::<T, D> {
        seed: 0,
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        populations: 1,
        population_size: 4,
        niterations: 1,
        ncycles_per_iteration: 1,
        maxsize: 10,
        maxdepth: 10,
        migration: false,
        hof_migration: false,
        optimizer_probability: 0.0,
        use_baseline: false,
        ..Default::default()
    };

    let add = <TestOps as HasOp<builtin::Add>>::op_id();
    let f_tree = dynamic_expressions::PostfixExpr::<T, TestOps, D>::new(
        vec![
            PNode::Var { feature: 0u16 },
            PNode::Var { feature: 1u16 },
            PNode::Op { arity: 2, op: add.id },
        ],
        vec![],
        Default::default(),
    );
    let g_tree = dynamic_expressions::PostfixExpr::<T, TestOps, D>::new(
        vec![PNode::Var { feature: 0u16 }],
        vec![],
        Default::default(),
    );

    let structure = TemplateStructure::<T, TestOps, D>::new(vec![("f", 2), ("g", 1)], vec![("a", 1)], |ctx, x| {
        let f = ctx.call("f", &[x[0], x[1]]);
        let g = ctx.call("g", &[x[2]]);
        if !f.valid || !g.valid {
            return ValidVec {
                x: vec![T::nan(); ctx.n_rows()],
                valid: false,
            };
        }
        let a0 = ctx.param("a").unwrap()[0];
        let mut out = vec![T::zero(); ctx.n_rows()];
        for (outv, (&fv, &gv)) in out.iter_mut().zip(f.x.iter().zip(g.x.iter())) {
            *outv = fv + a0 * gv;
        }
        ValidVec { x: out, valid: true }
    });

    let template = TemplateExpression::<T, TestOps, D> {
        structure: std::sync::Arc::new(structure),
        trees: vec![f_tree, g_tree],
        params: vec![ParamVector::new(vec![2.0])],
    };

    let spec = TemplateSpec::new(template.structure.clone());
    let _ = spec; // smoke test: spec is constructible and Clone.

    let mut member = PopMember::from_expr(MemberId(0), None, 0, template, dataset.n_features);
    let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
    let tagged = crate::TaggedDataset::new(&dataset, None);

    assert!(member.evaluate(&tagged, &options, &mut evaluator));
    assert!(member.loss.is_finite());
    assert!(member.loss < 1e-10);
    assert!(member.expr.check_constraints(&options, options.maxsize));

    // Plan rebuild smoke test.
    member.plan = member.expr.build_plan(dataset.n_features);
}
