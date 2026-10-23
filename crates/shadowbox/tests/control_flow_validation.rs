//! Every project-owned control-flow program must validate before and after
//! optimization. Unsupported instructions fail this gate.
#[path = "support/control_flow_inputs.rs"]
mod control_flow_inputs;
#[path = "support/control_flow_validation.rs"]
mod control_flow_validation;
#[path = "support/fuzz_inputs.rs"]
mod fuzz_inputs;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

#[test]
fn structured_shapes_validate_before_and_after_optimization() {
    let state = fuzz_inputs::graphics_state();
    let headers = fuzz_inputs::fuzz_headers();
    let mut rng = fuzz_inputs::Rng(0x5eed_1234);
    for seed in (0..32).chain((0..16).map(|_| rng.next())) {
        for (name, code) in control_flow_inputs::programs(seed) {
            for (stage, header) in headers.iter().enumerate() {
                let case = format!("{name} seed {seed:016x} stage {stage}");
                control_flow_validation::check(&case, &code, header, &state);
            }
        }
    }
    // Keep the straight-line fuzz bundle in this gate as well.
    // Mesa nak/sm50.rs:3178-3187 OpNop, CC.T at 8..12;
    // :225-235 PT predicate at 16..19.
    let nop = 0x50b0 << 48 | 7 << 16 | 15 << 8;
    control_flow_validation::check(
        "nop_bundle",
        &fuzz_inputs::fuzz_program(nop),
        &headers[0],
        &state,
    );
}

#[test]
fn decoder_has_exactly_the_two_permitted_nop_arms() {
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}
