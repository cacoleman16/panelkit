//! Deterministic, splittable PRNG — hand-rolled so the core has no `rand`
//! dependency and so the inference layer can give every bootstrap replicate its
//! own independent, seed-derived substream (this is what makes parallel results
//! bit-identical regardless of thread count).
//!
//! `SplitMix64` seeds the state of `Xoshiro256++`, the actual generator.

/// SplitMix64 — used purely to expand a single `u64` seed into the four words
/// of Xoshiro state, and to derive child seeds.
#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Xoshiro256++ — fast, high-quality, with a `jump` for stream separation.
#[derive(Clone)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// Seed from a single `u64` (expanded via SplitMix64).
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()];
        // Guard against the all-zero state.
        if s == [0, 0, 0, 0] {
            return Xoshiro256pp { s: [1, 2, 3, 4] };
        }
        Xoshiro256pp { s }
    }

    /// Deterministically derive an independent substream for index `i`.
    ///
    /// Used by the inference driver: replicate `i` always gets the same stream,
    /// so the resampling result is independent of scheduling / thread count.
    pub fn substream(master_seed: u64, i: u64) -> Self {
        // Mix the master seed with the index through SplitMix64 so different
        // (seed, i) pairs give well-separated Xoshiro states.
        let mut sm = SplitMix64::new(master_seed ^ i.wrapping_mul(0xD1B5_4A32_D192_ED03));
        let mixed = sm.next_u64();
        Xoshiro256pp::seed_from_u64(mixed)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)` using the top 53 bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)` via Lemire's rejection-free multiply-shift
    /// (with a small rejection step for exact uniformity).
    pub fn gen_range(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        let n = n as u64;
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut lo = m as u64;
        if lo < n {
            let t = n.wrapping_neg() % n;
            while lo < t {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                lo = m as u64;
            }
        }
        (m >> 64) as usize
    }

    /// Standard normal via the Box–Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        // Guard u1 away from 0 to avoid ln(0).
        let u1 = (self.next_f64()).max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// Fisher–Yates in-place shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.gen_range(i + 1);
            slice.swap(i, j);
        }
    }
}
