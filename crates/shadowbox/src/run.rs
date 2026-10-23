//! What a run is. The host program says it once, at its own entry (its
//! options or environment are its edge), and passes the run in; the run
//! then travels its own track: a quiet run formats nothing, a traced run
//! writes its stack-proof diagnostics to stderr. Every pass that reports is
//! generic over [`Run`]; the library reads no environment.

use crate::brx::Track;
use crate::decoder::{Quiet, StackTrace, Traced};
use crate::ir::{Instruction, InstructionKind};
use std::io::Write;

/// A run, chosen by its host. Only [`QuietRun`] and [`TracedRun`] are
/// runs: the behavior lives in the crate.
#[allow(private_bounds)]
pub trait Run: Explain {}

impl<T: Explain> Run for T {}

/// The behavior of a run's diagnostics.
pub(crate) trait Explain: Copy + Send + Sync {
    /// The diagnostics of one stack-proof exploration on this run.
    type Stack: StackTrace;
    /// A new exploration's diagnostics.
    fn stack(self) -> Self::Stack;
    /// The block proof found `x`'s guard PT or !PT, pruning one history.
    fn pruned(x: &Instruction);
    /// The linear fallback resolves the pop `x` to `previous`.
    fn fallback_resolves(self, x: &Instruction, previous: InstructionKind);
    /// The linear fallback also leaves the pop `x` unresolved.
    fn fallback_leaves(self, x: &Instruction);
    /// Resolve the register BRX tables, `resolve` writing the proof's
    /// diagnostics where this run sends them.
    fn brx(self, resolve: impl FnOnce(&mut dyn Write) -> Track) -> Track;
}

/// A run nobody asked to explain itself.
#[derive(Clone, Copy, Debug)]
pub struct QuietRun;

/// A run whose stack proofs explain themselves on stderr. Hosts choose it
/// from their own edge (the CLI and the novena host from
/// `SHADOWBOX_TRACE_STACK=1`).
#[derive(Clone, Copy, Debug)]
pub struct TracedRun;

impl Explain for QuietRun {
    type Stack = Quiet;
    fn stack(self) -> Quiet {
        Quiet
    }
    fn pruned(_: &Instruction) {}
    fn fallback_resolves(self, _: &Instruction, _: InstructionKind) {}
    fn fallback_leaves(self, _: &Instruction) {}
    fn brx(self, resolve: impl FnOnce(&mut dyn Write) -> Track) -> Track {
        resolve(&mut std::io::sink())
    }
}

impl Explain for TracedRun {
    type Stack = Traced<std::io::StderrLock<'static>>;
    fn stack(self) -> Self::Stack {
        Traced::new(std::io::stderr().lock())
    }
    fn pruned(x: &Instruction) {
        crate::decoder::trace_predicate_pruning(&mut Some(&mut std::io::stderr().lock()), x, true);
    }
    fn fallback_resolves(self, x: &Instruction, previous: InstructionKind) {
        // Diagnostics must not change translation, even if stderr is closed.
        let _ = writeln!(
            std::io::stderr().lock(),
            "SHADOWBOX_TRACE_STACK: main linear fallback resolves pop at byte 0x{:x}: {:?}",
            x.address,
            previous
        );
    }
    fn fallback_leaves(self, x: &Instruction) {
        let _ = writeln!(
            std::io::stderr().lock(),
            "SHADOWBOX_TRACE_STACK: main linear fallback also leaves pop at byte 0x{:x} unresolved",
            x.address
        );
    }
    fn brx(self, resolve: impl FnOnce(&mut dyn Write) -> Track) -> Track {
        let mut output = std::io::stderr().lock();
        let track = resolve(&mut output);
        crate::brx::write_unlearned(&mut output, &track);
        track
    }
}
