//! An instruction's guard, as what its guard field is. The public
//! instruction carries the field as it arrived; this station reads it once
//! per pass and hands the pass the guard as its own type. Each pass's guard
//! behavior is a trait the guard types implement; [`Guard`] gathers them.
//!
//! PT is P7, always true; !PT never runs (envytools gm107.c:363-364,408-411;
//! Mesa sm50.rs:225-235). An explicit PT guard differs from no guard field:
//! discovery and push placement keep it predicated.

use crate::ir::Predicate;

/// No guard field: the instruction always runs.
#[derive(Clone, Copy)]
pub(crate) struct Unguarded(());

/// An explicit PT guard: always true, yet present.
#[derive(Clone, Copy)]
pub(crate) struct Pt(());

/// !PT: the instruction never runs.
#[derive(Clone, Copy)]
pub(crate) struct NotPt(());

/// A guard on a real predicate register, P0..P6, in either sense.
#[derive(Clone, Copy)]
pub(crate) struct When(Predicate);

/// A guard field that is present: the predicate it reads.
pub(crate) trait Field: Copy {
    fn predicate(self) -> Predicate;
}

impl Field for Pt {
    fn predicate(self) -> Predicate {
        Predicate {
            register: 7,
            inverted: false,
        }
    }
}

impl Field for NotPt {
    fn predicate(self) -> Predicate {
        Predicate {
            register: 7,
            inverted: true,
        }
    }
}

impl Field for When {
    fn predicate(self) -> Predicate {
        self.0
    }
}

/// What a guard does in every pass that meets one.
pub(crate) trait Guard:
    Copy
    + Send
    + Sync
    + 'static
    + crate::stack::BlockGuard
    + crate::decoder::FallGuard
    + crate::decoder::ExploreGuard
    + crate::structure::EdgeGuard
    + crate::flow::Stands
{
}
impl<G> Guard for G where
    G: Copy
        + Send
        + Sync
        + 'static
        + crate::stack::BlockGuard
        + crate::decoder::FallGuard
        + crate::decoder::ExploreGuard
        + crate::structure::EdgeGuard
        + crate::flow::Stands
{
}

/// What a pass does with the guard once it is its own type.
pub(crate) trait Arrive {
    type Out;
    fn arrive(self, guard: impl Guard) -> Self::Out;
}

/// The station: the guard field arrives, and `pass` meets the guard it forms.
pub(crate) fn arrive<A: Arrive>(field: Option<Predicate>, pass: A) -> A::Out {
    match field {
        None => pass.arrive(Unguarded(())),
        Some(Predicate {
            register: 7,
            inverted: false,
        }) => pass.arrive(Pt(())),
        Some(Predicate {
            register: 7,
            inverted: true,
        }) => pass.arrive(NotPt(())),
        Some(p) => pass.arrive(When(p)),
    }
}
