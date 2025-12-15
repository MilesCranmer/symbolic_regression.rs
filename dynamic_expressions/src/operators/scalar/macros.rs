#[macro_export]
#[doc(hidden)]
macro_rules! __default_op_str {
    (Add) => {
        "+"
    };
    (Sub) => {
        "-"
    };
    (Mul) => {
        "*"
    };
    (Div) => {
        "/"
    };
    (Neg) => {
        "-"
    };
    ($other:ident) => {
        $crate::paste::paste! { stringify!([<$other:snake>]) }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! define_scalar_ops {
    (
        $vis:vis struct $Ops:ident<$t:ty>;
        ops {
            $(($arity:literal, $enum_name:ident) {
                $($op_name:ident => ($op_eval:path, $op_partial:path),)*
            })*
        }
    ) => {
        #[derive(Copy, Clone, Debug, Default)]
        $vis struct $Ops;

        $(
            #[repr(u16)]
            #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
            $vis enum $enum_name { $($op_name,)* }
        )*

        impl $crate::operators::scalar::ScalarOpSet<$t> for $Ops {
            fn eval(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::EvalKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::eval_nary::<$arity, $t>($op_eval, ctx.out, ctx.args, ctx.opts),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn diff(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::DiffKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::diff_nary::<$arity, $t>(
                                        $op_eval,
                                        $op_partial,
                                        ctx.out_val,
                                        ctx.out_der,
                                        ctx.args,
                                        ctx.dargs,
                                        ctx.opts,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn grad(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::GradKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::grad_nary::<$arity, $t>(
                                        $op_eval,
                                        $op_partial,
                                        ctx,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }
        }

        $(
            $(
                impl $crate::operators::scalar::HasOp<$crate::operators::builtin::$op_name, $arity>
                    for $Ops
                {
                    const ID: u16 = $enum_name::$op_name as u16;
                }
            )*
        )*

        impl $crate::operators::names::OpNames for $Ops {
            fn op_name(op: $crate::operators::scalar::OpId) -> &'static str {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) => $crate::__default_op_str!($op_name),
                            )*
                            _ => "unknown_op",
                        },
                    )*
                    _ => "unknown_op",
                }
            }
        }

    };
}

#[macro_export]
macro_rules! opset {
    (
        $vis:vis struct $Ops:ident<$t:ty>;
        ops {
            $(($arity:literal, $enum_name:ident) { $($op_name:ident,)* })*
        }
    ) => {
        #[derive(Copy, Clone, Debug, Default)]
        $vis struct $Ops;

        $(
            #[repr(u16)]
            #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
            $vis enum $enum_name { $($op_name,)* }
        )*

        impl $crate::operators::scalar::ScalarOpSet<$t> for $Ops {
            fn eval(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::EvalKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::eval_apply::<$arity, $t, $crate::operators::builtin::$op_name>(
                                        ctx.out, ctx.args, ctx.opts,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn diff(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::DiffKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::diff_apply::<$arity, $t, $crate::operators::builtin::$op_name>(
                                        ctx.out_val, ctx.out_der, ctx.args, ctx.dargs, ctx.opts,
                                    ),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }

            fn grad(
                op: $crate::operators::scalar::OpId,
                ctx: $crate::operators::scalar::GradKernelCtx<'_, '_, $t>,
            ) -> bool {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) =>
                                    $crate::operators::scalar::grad_apply::<$arity, $t, $crate::operators::builtin::$op_name>(ctx),
                            )*
                            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
                        },
                    )*
                    _ => panic!("unsupported arity {}", op.arity),
                }
            }
        }

        $(
            $(
                impl $crate::operators::scalar::HasOp<$crate::operators::builtin::$op_name, $arity> for $Ops {
                    const ID: u16 = $enum_name::$op_name as u16;
                }
            )*
        )*

        impl $crate::operators::names::OpNames for $Ops {
            fn op_name(op: $crate::operators::scalar::OpId) -> &'static str {
                match op.arity {
                    $(
                        $arity => match op.id {
                            $(
                                x if x == ($enum_name::$op_name as u16) => {
                                    <$crate::operators::builtin::$op_name as $crate::operators::builtin::BuiltinOp<$t, $arity>>::DISPLAY
                                }
                            )*
                            _ => "unknown_op",
                        },
                    )*
                    _ => "unknown_op",
                }
            }
        }

        impl $Ops {
            pub const REGISTRY: &'static [$crate::operators::registry::OpInfo] = &[
                $(
                    $(
                        $crate::operators::registry::OpInfo {
                            op: $crate::operators::scalar::OpId {
                                arity: $arity as u8,
                                id: $enum_name::$op_name as u16,
                            },
                            name: <$crate::operators::builtin::$op_name as $crate::operators::builtin::BuiltinOp<$t, $arity>>::NAME,
                            display: <$crate::operators::builtin::$op_name as $crate::operators::builtin::BuiltinOp<$t, $arity>>::DISPLAY,
                            infix: <$crate::operators::builtin::$op_name as $crate::operators::builtin::BuiltinOp<$t, $arity>>::INFIX,
                            commutative: <$crate::operators::builtin::$op_name as $crate::operators::builtin::OpMeta<$arity>>::COMMUTATIVE,
                            associative: <$crate::operators::builtin::$op_name as $crate::operators::builtin::OpMeta<$arity>>::ASSOCIATIVE,
                            complexity: <$crate::operators::builtin::$op_name as $crate::operators::builtin::OpMeta<$arity>>::COMPLEXITY,
                        },
                    )*
                )*
            ];
        }

        impl $crate::operators::registry::OpRegistry for $Ops {
            #[inline]
            fn registry() -> &'static [$crate::operators::registry::OpInfo] {
                Self::REGISTRY
            }
        }
    };
}
