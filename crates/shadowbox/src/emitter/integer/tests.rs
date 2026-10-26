use super::*;

/// Evaluate the emitted arithmetic, rather than a second copy of the lowering.
/// Opcode behavior is SPIR-V 3.3.13-3.3.15. Oversized shifts/extracts fail
/// even when an OpSelect would discard their result.
fn evaluate(words: &[u32], mut values: HashMap<u32, u32>, result: u32) -> u32 {
    let mut at = 0;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        let a = &words[at + 1..at + count];
        let v = |i: usize| values[&a[i]];
        let value = match op {
            Op::IAdd => v(2).wrapping_add(v(3)),
            Op::ISub => v(2).wrapping_sub(v(3)),
            Op::LogicalNotEqual | Op::BitwiseXor => v(2) ^ v(3),
            Op::LogicalOr | Op::BitwiseOr => v(2) | v(3),
            Op::LogicalAnd | Op::BitwiseAnd => v(2) & v(3),
            Op::LogicalNot => u32::from(v(2) == 0),
            Op::Select => {
                if v(2) != 0 {
                    v(3)
                } else {
                    v(4)
                }
            }
            Op::IEqual => u32::from(v(2) == v(3)),
            Op::INotEqual => u32::from(v(2) != v(3)),
            Op::UGreaterThan => u32::from(v(2) > v(3)),
            Op::SGreaterThan => u32::from((v(2) as i32) > (v(3) as i32)),
            Op::UGreaterThanEqual => u32::from(v(2) >= v(3)),
            Op::SGreaterThanEqual => u32::from((v(2) as i32) >= (v(3) as i32)),
            Op::ULessThan => u32::from(v(2) < v(3)),
            Op::SLessThan => u32::from((v(2) as i32) < (v(3) as i32)),
            Op::ULessThanEqual => u32::from(v(2) <= v(3)),
            Op::SLessThanEqual => u32::from((v(2) as i32) <= (v(3) as i32)),
            Op::ShiftRightLogical | Op::ShiftRightArithmetic | Op::ShiftLeftLogical => {
                assert!(v(3) < 32, "undefined SPIR-V shift: {}", v(3));
                match op {
                    Op::ShiftRightLogical => v(2) >> v(3),
                    Op::ShiftRightArithmetic => ((v(2) as i32) >> v(3)) as u32,
                    _ => v(2) << v(3),
                }
            }
            Op::Not => !v(2),
            Op::BitFieldSExtract | Op::BitFieldUExtract => {
                let (base, offset, bits) = (v(2), v(3), v(4));
                assert!(offset <= 32 && bits <= 32 - offset);
                if bits == 0 {
                    0
                } else {
                    let left = base << (32 - offset - bits);
                    if op == Op::BitFieldSExtract {
                        ((left as i32) >> (32 - bits)) as u32
                    } else {
                        left >> (32 - bits)
                    }
                }
            }
            Op::BitReverse => v(2).reverse_bits(),
            _ => panic!("unhandled arithmetic opcode {}", u32::from(op)),
        };
        assert!(values.insert(a[1], value).is_none(), "duplicate ID");
        at += count;
    }
    values[&result]
}

fn run(inputs: &[u32], lower: impl FnOnce(&mut IntegerEmitter<'_>, &[u32]) -> u32) -> u32 {
    let constants = (0..=64)
        .chain([255, 0xffff])
        .map(|v| (v, v + 10))
        .collect::<HashMap<_, _>>();
    let mut values = constants
        .iter()
        .map(|(&v, &id)| (id, v))
        .collect::<HashMap<_, _>>();
    let ids = inputs
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let id = 100_000 + i as u32;
            values.insert(id, v);
            id
        })
        .collect::<Vec<_>>();
    let mut words = Vec::new();
    let mut id = 200_000;
    let result = lower(
        &mut IntegerEmitter {
            w: &mut words,
            id: &mut id,
            uint: 1,
            boolt: 2,
            constants: &constants,
        },
        &ids,
    );
    evaluate(&words, values, result)
}

#[test]
fn funnel_matches_mesa_maxwell_fold_at_every_boundary() {
    // Independent 64-bit reference: Mesa nak/ir.rs:4501-4543.
    let pairs = [
        (0, 0),
        (0x7654_3210, 0xfedc_ba98),
        (u32::MAX, u32::MAX),
        (1, 0x8000_0000),
    ];
    for (low, high) in pairs {
        for ty in [
            FunnelType::Bits32,
            FunnelType::Unsigned64,
            FunnelType::Signed64,
        ] {
            for wrap in [false, true] {
                for right in [false, true] {
                    for dst_high in [false, true] {
                        if !right && !dst_high {
                            continue;
                        }
                        for count in [0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 255, u32::MAX] {
                            let width = if ty == FunnelType::Bits32 { 32 } else { 64 };
                            let shift = if wrap {
                                count & (width - 1)
                            } else {
                                count.min(width)
                            };
                            let pair = (u64::from(high) << 32) | u64::from(low);
                            let shifted = if right {
                                if ty == FunnelType::Signed64 {
                                    (pair as i64)
                                        .checked_shr(shift)
                                        .unwrap_or((pair as i64) >> 63)
                                        as u64
                                } else {
                                    pair.checked_shr(shift).unwrap_or(0)
                                }
                            } else {
                                pair.checked_shl(shift).unwrap_or(0)
                            };
                            let expected = if dst_high {
                                (shifted >> 32) as u32
                            } else {
                                shifted as u32
                            };
                            let actual = run(&[low, high, count], |e, v| {
                                e.funnel(v[0], v[1], v[2], right, dst_high, ty, wrap)
                            });
                            assert_eq!(
                                actual, expected,
                                "{ty:?} right={right} hi={dst_high} wrap={wrap} count={count}"
                            );
                            let immediate = run(&[low, high], |e, v| {
                                e.funnel_immediate(v[0], v[1], count, right, dst_high, ty, wrap)
                            });
                            assert_eq!(immediate, expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn lea_hi_negates_the_whole_shifted_pair() {
    // Ryujinx InstEmitIntegerArithmetic.cs:493-519, tested using native u64.
    for (low, high) in [
        (0, 0),
        (0, 0xffff_ffff),
        (0x8000_0000, 0),
        (1, 0x8123_4567),
        (u32::MAX, u32::MAX),
    ] {
        for shift in 0..32u8 {
            for negate in [false, true] {
                for addend in [0, 17, u32::MAX] {
                    let pair = ((u64::from(high) << 32) | u64::from(low)) << shift;
                    let pair = if negate { pair.wrapping_neg() } else { pair };
                    let expected = ((pair >> 32) as u32).wrapping_add(addend);
                    let actual = run(&[low, high, addend], |e, v| {
                        e.lea_hi(v[0], v[1], v[2], shift, negate)
                    });
                    assert_eq!(actual, expected, "shift={shift} negate={negate}");
                }
            }
        }
    }
}

#[test]
fn add3_halves_negations_and_partial_sum_shift() {
    // Ryujinx InstEmitIntegerArithmetic.cs:357-404.
    let parts = [HalfSelect::Full, HalfSelect::Low, HalfSelect::High];
    for a in parts {
        for b in parts {
            for c in parts {
                let halves = [a, b, c];
                for shift in [
                    PartialSumShift::Unshifted,
                    PartialSumShift::Left,
                    PartialSumShift::Right,
                ] {
                    for neg in 0..8 {
                        let negate = [neg & 1 != 0, neg & 2 != 0, neg & 4 != 0];
                        let inputs: [u32; 3] = [0xffff_abcd, 0x8123_4567, 0xcdef_9876];
                        let mut operands = inputs;
                        for i in 0..3 {
                            operands[i] = match halves[i] {
                                HalfSelect::Full => operands[i],
                                HalfSelect::Low => operands[i] & 0xffff,
                                HalfSelect::High => operands[i] >> 16,
                            };
                            if negate[i] {
                                operands[i] = operands[i].wrapping_neg();
                            }
                        }
                        let ab = operands[0].wrapping_add(operands[1]);
                        let ab = match shift {
                            PartialSumShift::Unshifted => ab,
                            PartialSumShift::Left => ab << 16,
                            PartialSumShift::Right => ab >> 16,
                        };
                        let actual = run(&inputs, |e, v| {
                            e.add3([v[0], v[1], v[2]], halves, shift, negate)
                        });
                        assert_eq!(actual, ab.wrapping_add(operands[2]));
                    }
                }
            }
        }
    }
}

#[test]
fn bfe_signedness_reversal_zero_and_full_width() {
    // Ryujinx InstEmitBitfield.cs:136-149 and SPIR-V 3.3.14.
    for base in [0, 0xfedc_ba98u32, 0x8000_0001, u32::MAX] {
        for offset in 0..=32 {
            for bits in 0..=32 - offset {
                for signed in [false, true] {
                    for reverse in [false, true] {
                        let input = if reverse { base.reverse_bits() } else { base };
                        let mask = if bits == 32 {
                            u32::MAX
                        } else {
                            (1 << bits) - 1
                        };
                        let field = input.checked_shr(offset).unwrap_or(0) & mask;
                        let expected = if signed && bits > 0 && (field & (1 << (bits - 1))) != 0 {
                            field | !mask
                        } else {
                            field
                        };
                        let actual = run(&[base, offset | (bits << 8)], |e, v| {
                            e.bfe(v[0], v[1], signed, reverse)
                        });
                        assert_eq!(actual, expected);
                        let immediate = run(&[base], |e, v| {
                            e.bfe_immediate(v[0], offset | (bits << 8), signed, reverse)
                        });
                        assert_eq!(immediate, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn extended_iset_matches_64_bit_comparisons() {
    // Ryujinx InstEmitIntegerComparison.cs:207-266. Z and C represent
    // equality and no borrow in the preceding low-word comparison.
    for signed in [false, true] {
        let signedness = if signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };
        for (a, b) in [
            (0, 0),
            (5, 5),
            (1, 0),
            (0, 1),
            (0x8000_0000, 0x7fff_ffff),
            (u32::MAX, 0),
        ] {
            for (lo_a, lo_b) in [(0, 0), (1, 0), (0, 1), (u32::MAX, 1)] {
                let pair_a = (u64::from(a) << 32) | u64::from(lo_a);
                let pair_b = (u64::from(b) << 32) | u64::from(lo_b);
                let order = if signed {
                    (pair_a as i64).cmp(&(pair_b as i64))
                } else {
                    pair_a.cmp(&pair_b)
                };
                for compare in [
                    Relation::Lt,
                    Relation::Le,
                    Relation::Gt,
                    Relation::Ge,
                    Relation::Eq,
                    Relation::Ne,
                ] {
                    let expected = match compare {
                        Relation::Lt => order.is_lt(),
                        Relation::Le => !order.is_gt(),
                        Relation::Gt => order.is_gt(),
                        Relation::Ge => !order.is_lt(),
                        Relation::Eq => order.is_eq(),
                        Relation::Ne => !order.is_eq(),
                    };
                    let actual = run(
                        &[a, b, u32::from(lo_a == lo_b), u32::from(lo_a >= lo_b)],
                        |e, v| e.extended_compare(compare, signedness, v[0], v[1], v[2], v[3]),
                    );
                    assert_eq!(actual, u32::from(expected));
                }
            }
        }
    }
}
