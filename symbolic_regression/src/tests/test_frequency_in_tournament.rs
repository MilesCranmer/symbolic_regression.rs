use super::common::{TestOps, D, T};
use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::member::{MemberId, PopMember};
use crate::operator_library::OperatorLibrary;
use crate::population::Population;
use crate::selection::best_of_sample;
use crate::Options;
use dynamic_expressions::expr::{Metadata, PNode, PostfixExpr};
use num_traits::One;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn leaf_expr() -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(
        vec![PNode::Var { feature: 0 }],
        Vec::new(),
        Metadata::default(),
    )
}

#[test]
fn tournament_penalizes_frequent_sizes_when_enabled() {
    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        tournament_selection_n: 2,
        tournament_selection_p: 1.0,
        use_frequency_in_tournament: true,
        adaptive_parsimony_scaling: 100.0,
        ..Default::default()
    };
    let mut rng = StdRng::seed_from_u64(0);

    let mut a = PopMember::from_expr(MemberId(1), None, 0, leaf_expr(), 1);
    let mut b = PopMember::from_expr(MemberId(2), None, 0, leaf_expr(), 1);
    a.complexity = 1;
    b.complexity = 3;
    a.cost = T::one();
    b.cost = T::one();

    let pop = Population::new(vec![a, b]);

    let mut stats = RunningSearchStatistics::new(options.maxsize, 1000);
    // Artificially make size=1 very frequent and size=3 rare.
    for _ in 0..100 {
        stats.update_frequencies(1);
    }
    stats.update_frequencies(3);
    stats.normalize();

    let chosen = best_of_sample::<T, TestOps, D, _>(&mut rng, &pop, &stats, &options);

    // If frequency is used in tournament, the rare (larger) member should be favored here.
    assert_eq!(chosen.id, MemberId(2));
}
