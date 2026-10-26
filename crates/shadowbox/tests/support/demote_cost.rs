//! Project-owned discard-heavy fixtures and timestamp experiment.
//! Semantics and measurement boundaries: docs/reference-sources.md
use super::*;
use shadowbox::spirv::Op;

fn terminating(words: &[u32]) -> Vec<u32> {
    // Old mapping: Ryujinx CodeGen/Spirv/Instructions.cs:506-509.
    // Replace the demote block's branch with a terminating OpKill.
    let mut result = words[..5].to_vec();
    let mut at = 5;
    let mut replaced = 0;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        assert_ne!(
            op,
            Op::IsHelperInvocationEXT,
            "fixture must not query helper state"
        );
        if op == Op::DemoteToHelperInvocation {
            assert_eq!(count, 1);
            result.push(1 << 16 | u32::from(Op::Kill));
            at += count;
            assert_eq!(
                Op::of(words[at]),
                Op::Branch,
                "demote block must end in a branch"
            );
            at += (words[at] >> 16) as usize;
            replaced += 1;
        } else {
            result.extend_from_slice(&words[at..at + count]);
            at += count;
        }
    }
    assert_eq!(replaced, 1);
    result
}

fn fixture(length: usize, position: usize) -> Vec<u64> {
    // ALD, MOV and FSETP field builders are shared with divergent_graphics.
    let mut xs = divergent_graphics::setup(192.0);
    // Initialize a live recurrence from the fragment's x coordinate.
    xs.push(0x5c98 << 48 | 15 << 39 | 10 << 20 | PT);
    xs.push(mov(14, 1.0001f32.to_bits()));
    xs.push(mov(15, 0.001f32.to_bits()));
    for i in 0..=length {
        if i == position {
            xs.push(divergent_graphics::flow(0xe330, 0, false));
        }
        if i < length {
            // Mesa sm50.rs:681-704 FMUL register; :481-539 FADD register.
            // Ryujinx InstEmitFloatArithmetic.cs:116-124,211-219: FP32 math.
            xs.push(0x5c68 << 48 | 14 << 20 | PT);
            xs.push(0x5c58 << 48 | 15 << 20 | PT);
        }
    }
    xs
}

fn quantiles(mut samples: Vec<f64>) -> [f64; 3] {
    samples.sort_by(f64::total_cmp);
    [
        samples[samples.len() / 10],
        samples[samples.len() / 2],
        samples[samples.len() * 9 / 10],
    ]
}

#[test]
#[ignore = "GPU timestamp cost experiment"]
fn gpu_demote_cost() {
    let suite = Suite::new("demote-cost");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    let mut gpu = Graphics::new();
    let input: Vec<_> = [[-1., -1.], [3., -1.], [-1., 3.]]
        .map(|p| [p[0], p[1], 0.25, 1., 0., 0., 0., 0.])
        .into();
    let h = header(Stage::Fragment);
    let mut rows = String::from("math_instructions\tposition\tkill_p10_us\tkill_median_us\tkill_p90_us\tdemote_p10_us\tdemote_median_us\tdemote_p90_us\tdelta_percent\n");
    for length in [16, 64, 256] {
        for (name, position) in [("early", 0), ("mid", length / 2), ("late", length)] {
            let raw = code(&fixture(length, position));
            let translated = Result::from(translate_with_graphics_state(
                shadowbox::QuietRun,
                &ShaderInput {
                    code: &raw,
                    program_record: &h,
                },
                &GraphicsState::default(),
            ))
            .unwrap();
            assert!(translated.requires_demote_to_helper_invocation);
            let demote = translated.spirv;
            let kill = terminating(&demote);
            for (name, words) in [("demote", &demote), ("kill", &kill)] {
                let path = suite.scratch.join(format!("{name}.spv"));
                fs::write(
                    &path,
                    words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let result = Command::new("spirv-val")
                    .args(["--target-env", "vulkan1.2"])
                    .arg(&path)
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
            let shaders = [
                vec![(Stage::Vertex, producer.clone()), (Stage::Fragment, kill)],
                vec![(Stage::Vertex, producer.clone()), (Stage::Fragment, demote)],
            ];
            let draws = shaders.each_ref().map(|s| Draw {
                shaders: s,
                vertices: &input,
                vertex_count: 3,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                samples: vk::SampleCountFlags::TYPE_1,
            });
            assert_eq!(
                gpu.run_timed(&draws[0], 1, 1).0,
                gpu.run_timed(&draws[1], 1, 1).0
            );
            let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            // Alternate five-sample batches. Each fresh pipeline gets two warmups.
            for batch in 0..9 {
                for offset in 0..2 {
                    let variant = (batch + offset) % 2;
                    let (_, times) = gpu.run_timed(&draws[variant], 8, 7);
                    samples[variant].extend_from_slice(&times[2..]);
                }
            }
            let [k, d] = samples.map(quantiles);
            rows.push_str(&format!(
                "{}\t{name}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}\n",
                length * 2,
                k[0],
                k[1],
                k[2],
                d[0],
                d[1],
                d[2],
                (d[1] / k[1] - 1.0) * 100.0
            ));
        }
    }
    fs::write(suite.report.join("aggregate.tsv"), &rows).unwrap();
    print!("{rows}");
    fs::remove_dir_all(suite.scratch).unwrap();
}
