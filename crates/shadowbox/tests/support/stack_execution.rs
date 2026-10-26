//! Execute the scalar SPIR-V subset emitted by the stack-jump regressions.
//! This checks each lane separately. Barrier counts check placement, not
//! workgroup rendezvous. spirv-val checks the modules independently.
use shadowbox::spirv::Op;
use std::collections::HashMap;

pub fn run(words: &[u32]) -> (u32, usize) {
    run_with_constants(words, &[])
}

pub fn run_with_constants(words: &[u32], buffers: &[(u32, &[u32])]) -> (u32, usize) {
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        instructions.push((Op::of(words[at]), &words[at + 1..at + count]));
        at += count;
    }
    let labels: HashMap<_, _> = instructions
        .iter()
        .enumerate()
        .filter(|(_, (op, _))| *op == Op::Label)
        .map(|(at, (_, args))| (args[0], at))
        .collect();
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut memory: HashMap<(u32, Vec<u32>), Vec<u32>> = HashMap::new();
    let mut pointers: HashMap<u32, (u32, Vec<u32>)> = HashMap::new();
    // Descriptor Binding decoration and Uniform storage: Mesa src/compiler/spirv/spirv.h:225,537
    // defines StorageClass Uniform=2 and Decoration Binding=33.
    let bindings: HashMap<_, _> = instructions
        .iter()
        .filter(|(op, a)| *op == Op::Decorate && a[1] == 33)
        .map(|(_, a)| (a[0], a[2]))
        .collect();
    let mut shared = None;
    let mut barriers = 0;
    let mut current = 0;
    let mut previous = 0;
    at = 0;
    for _ in 0..1_000_000 {
        let (op, a) = instructions[at];
        let get = |i: usize| values[&a[i]][0];
        let result = match op {
            // GLSL.std.450 UMin38: Mesa src/compiler/spirv/GLSL.std.450.h:61.
            Op::ExtInst => {
                assert_eq!(a[3], 38);
                Some(vec![get(4).min(get(5))])
            }
            Op::ConstantTrue => Some(vec![1]),
            Op::ConstantFalse => Some(vec![0]),
            Op::Constant => Some(vec![a[2]]),
            Op::Variable => {
                // Workgroup storage class: SPIR-V core grammar :13763-13765.
                if a[2] == 4 {
                    shared = Some(a[1]);
                }
                if a[2] == 2 {
                    let binding = bindings[&a[1]];
                    let data = buffers
                        .iter()
                        .find(|(b, _)| *b == binding)
                        .expect("reference execution requires the bound constant buffer")
                        .1;
                    for (row, words) in data.chunks(4).enumerate() {
                        let mut value = vec![0; 4];
                        value[..words.len()].copy_from_slice(words);
                        memory.insert((a[1], vec![0, row as u32]), value);
                    }
                }
                pointers.insert(a[1], (a[1], Vec::new()));
                None
            }
            Op::Load => Some(memory[&pointers[&a[2]]].clone()),
            Op::Store => {
                memory.insert(pointers[&a[0]].clone(), values[&a[1]].clone());
                None
            }
            Op::AccessChain => {
                // OpAccessChain adds element indices to its base pointer:
                // SPIR-V core grammar spirv.core.grammar.json:756-766.
                let (base, mut indices) = pointers[&a[2]].clone();
                indices.extend(a[3..].iter().map(|id| values[id][0]));
                pointers.insert(a[1], (base, indices));
                None
            }
            // Mesa src/compiler/spirv/spirv.h:1806: VectorExtractDynamic77.
            Op::VectorExtractDynamic => Some(vec![values[&a[2]][get(3) as usize]]),
            Op::CompositeExtract => Some(vec![values[&a[2]][a[3] as usize]]),
            Op::Bitcast => Some(vec![get(2)]),
            Op::IAdd => Some(vec![get(2).wrapping_add(get(3))]),
            Op::IAddCarry => {
                let sum = u64::from(get(2)) + u64::from(get(3));
                Some(vec![sum as u32, (sum >> 32) as u32])
            }
            Op::LogicalNotEqual | Op::INotEqual => Some(vec![u32::from(get(2) != get(3))]),
            Op::LogicalEqual | Op::IEqual => Some(vec![u32::from(get(2) == get(3))]),
            Op::LogicalOr => Some(vec![u32::from(get(2) != 0 || get(3) != 0)]),
            Op::LogicalAnd => Some(vec![u32::from(get(2) != 0 && get(3) != 0)]),
            Op::LogicalNot => Some(vec![u32::from(get(2) == 0)]),
            Op::Select => Some(vec![get(if get(2) != 0 { 3 } else { 4 })]),
            // Mesa src/compiler/spirv/spirv.h:1899: UGreaterThanEqual174.
            Op::UGreaterThanEqual => Some(vec![u32::from(get(2) >= get(3))]),
            Op::ULessThan => Some(vec![u32::from(get(2) < get(3))]),
            // OpShiftLeftLogical/BitwiseAnd: Mesa src/compiler/spirv/spirv.h:1919,1922.
            Op::ShiftLeftLogical => Some(vec![get(2) << get(3)]),
            Op::BitwiseAnd => Some(vec![get(2) & get(3)]),
            // OpShiftRightLogical: SPIR-V core grammar :2251-2263.
            Op::ShiftRightLogical => Some(vec![get(2) >> get(3)]),
            Op::ControlBarrier => {
                barriers += 1;
                None
            }
            Op::Label => {
                current = a[0];
                // OpPhi pairs are value/parent IDs: SPIR-V core grammar
                // spirv.core.grammar.json:2788-2796. Read every incoming
                // value before updating results, including loop-carried phis.
                let mut phis = Vec::new();
                while instructions.get(at + 1).is_some_and(|i| i.0 == Op::Phi) {
                    at += 1;
                    let phi = instructions[at].1;
                    let source = phi[2..]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .find(|pair| pair[1] == previous)
                        .expect("phi must have the executed predecessor")[0];
                    phis.push((phi[1], values[&source].clone()));
                }
                values.extend(phis);
                None
            }
            Op::Branch => {
                previous = current;
                at = labels[&a[0]];
                continue;
            }
            Op::BranchConditional => {
                previous = current;
                at = labels[&a[if get(0) != 0 { 1 } else { 2 }]];
                continue;
            }
            Op::Switch => {
                previous = current;
                let label = a[2..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .find(|pair| pair[0] == get(0))
                    .map_or(a[1], |pair| pair[1]);
                at = labels[&label];
                continue;
            }
            Op::Return => return (memory[&(shared.unwrap(), vec![0])][0], barriers),
            // Declarations, structured merge markers and labels.
            Op::ExtInstImport
            | Op::MemoryModel
            | Op::EntryPoint
            | Op::ExecutionMode
            | Op::Capability
            | Op::TypeVoid
            | Op::TypeBool
            | Op::TypeInt
            | Op::TypeFloat
            | Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeStruct
            | Op::TypeOpaque
            | Op::TypePointer
            | Op::TypeFunction
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
            | Op::TypeForwardPointer
            | Op::Function
            | Op::FunctionEnd
            | Op::Decorate
            | Op::MemberDecorate
            | Op::LoopMerge
            | Op::SelectionMerge => None,
            other => panic!("unexpected executable SPIR-V opcode {}", other.number()),
        };
        if let Some(result) = result {
            values.insert(a[1], result);
        }
        at += 1;
    }
    panic!("shader did not return before its trailing self-branch");
}
