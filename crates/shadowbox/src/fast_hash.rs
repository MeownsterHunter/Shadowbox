//! Small integer-key tables used during one translation. This original
//! rotate/xor/multiply implementation is FxHash-style, not cryptographic.
//! Table iteration must never determine emitted ID or declaration order.
use std::hash::{BuildHasherDefault, Hasher};

pub(crate) type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub(crate) type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FastHasher>>;

#[derive(Default)]
pub(crate) struct FastHasher(u64);
impl FastHasher {
    fn word(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}
impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, tail) = bytes.as_chunks::<8>();
        for &chunk in chunks {
            self.word(u64::from_le_bytes(chunk));
        }
        // An empty tail has no chunk; a partial one is exactly one.
        for tail in tail.chunks(8) {
            let mut word = [0; 8];
            word[..tail.len()].copy_from_slice(tail);
            self.word(u64::from_le_bytes(word));
        }
    }
    fn write_u8(&mut self, value: u8) {
        self.word(u64::from(value));
    }
    fn write_u16(&mut self, value: u16) {
        self.word(u64::from(value));
    }
    fn write_u32(&mut self, value: u32) {
        self.word(u64::from(value));
    }
    fn write_u64(&mut self, value: u64) {
        self.word(value);
    }
    fn write_usize(&mut self, value: usize) {
        self.word(value as u64);
    }
}
