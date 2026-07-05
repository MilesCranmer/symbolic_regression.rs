#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum PNode {
    Var { feature: u16 },
    Delay { feature: u16, offset: u16 },
    Const { idx: u16 },
    Op { arity: u8, op: u16 },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Src {
    Slot(u16),
    Var(u16),
    Delay { feature: u16, offset: u16 },
    Const(u16),
}
