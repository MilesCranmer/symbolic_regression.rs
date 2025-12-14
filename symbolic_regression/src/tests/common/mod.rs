use dynamic_expressions::opset;

pub type T = f64;
pub const D: usize = 2;

opset! {
    pub struct TestOps<f64>;
    ops {
        (1, UOps) { Sin, Cos, Exp, Log, Neg, }
        (2, BOps) { Add, Sub, Mul, Div, }
    }
}
