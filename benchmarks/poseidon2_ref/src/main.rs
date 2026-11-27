// Plonky3's Poseidon2-BabyBear/16, for two jobs the Triton kernel cannot do for itself:
//
//   kat       emit known-answer vectors as JSON (`data/vectors/poseidon2/bb16.json`)
//   bench     the tuned CPU rival, single-threaded: ns/permutation
//
// Usage:
//   cargo run --release -- kat   > ../../data/vectors/poseidon2/bb16.json
//   cargo run --release -- bench [n]
//
// `kat` vectors are the ones the numpy and Triton implementations are checked against;
// the inputs are fixed and small so they can be retyped if the JSON ever goes missing.
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};
use p3_field::{Field, PackedValue, PrimeField32};
use p3_symmetric::Permutation;
use std::time::Instant;

const Q: u32 = 2013265921;

fn vectors() -> Vec<[u32; 16]> {
    let mut cases: Vec<[u32; 16]> = Vec::new();
    // 1. the identity-ish ramp: 0..15
    let mut ramp = [0u32; 16];
    for (i, v) in ramp.iter_mut().enumerate() {
        *v = i as u32;
    }
    cases.push(ramp);
    // 2. offsets that stress the reduction paths, including q-1
    for off in [1u32, 97, 1000, 12_345, 1 << 30, Q - 1, 654_321] {
        let mut st = [0u32; 16];
        for (i, v) in st.iter_mut().enumerate() {
            let acc = off.wrapping_mul(i as u32 + 1).wrapping_add(7 * i as u32);
            *v = acc % Q;
        }
        cases.push(st);
    }
    // 3. all-max: the largest canonical state
    cases.push([Q - 1; 16]);
    // 4. deterministic pseudo-random states: the structured cases above exercise the edges,
    //    these exercise the middle (an LCG so the JSON is reproducible without shipping data)
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    for _ in 0..55 {
        let mut st = [0u32; 16];
        for v in st.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) % Q as u64) as u32;
        }
        cases.push(st);
    }
    cases
}

fn kat() {
    let perm = default_babybear_poseidon2_16();
    let cases = vectors();
    let mut out = String::from(
        "{\"width\":16,\"q\":2013265921,\"full_rounds\":8,\"partial_rounds\":13,\"source\":\"plonky3 0.8 p3-baby-bear default_babybear_poseidon2_16\",\"vectors\":[",
    );
    for (k, case) in cases.iter().enumerate() {
        let mut st = case.map(BabyBear::new);
        perm.permute_mut(&mut st);
        if k > 0 {
            out.push(',');
        }
        out.push_str("{\"in\":[");
        for (i, v) in case.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&v.to_string());
        }
        out.push_str("],\"out\":[");
        for (i, v) in st.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&v.as_canonical_u32().to_string());
        }
        out.push_str("]}");
    }
    out.push_str("]}");
    println!("{}", out);
}

fn bench(n: usize) {
    let perm = default_babybear_poseidon2_16();
    let mut state: [BabyBear; 16] = std::array::from_fn(|i| BabyBear::new(i as u32 + 1));
    // warm-up
    for _ in 0..10_000 {
        perm.permute_mut(&mut state);
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..n {
            perm.permute_mut(&mut state);
        }
        let dt = t0.elapsed().as_secs_f64() / n as f64;
        if dt < best {
            best = dt;
        }
    }
    // keep the compiler honest: the state must reach the output
    println!(
        "{{\"ns_per_permutation\":{:.3},\"state0\":{}}}",
        best * 1e9,
        state[0].as_canonical_u32()
    );
}

/// The path a real prover uses: `Packing::WIDTH` independent permutations per call, with the
/// monty-31x8 AVX2/AVX-512 vectorised field. A single scalar state (see `bench`) is
/// latency-bound and is *not* the rival to quote.
fn bench_packed(n: usize) {
    let perm = default_babybear_poseidon2_16();
    type P = <BabyBear as Field>::Packing;
    let lanes = P::WIDTH;
    let mut state: [P; 16] = std::array::from_fn(|i| P::from(BabyBear::new(i as u32 + 1)));
    for _ in 0..10_000 {
        perm.permute_mut(&mut state);
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..n {
            perm.permute_mut(&mut state);
        }
        let dt = t0.elapsed().as_secs_f64() / (n as f64 * lanes as f64);
        if dt < best {
            best = dt;
        }
    }
    println!(
        "{{\"lanes\":{},\"ns_per_permutation\":{:.3},\"mperm_per_s\":{:.2}}}",
        lanes,
        best * 1e9,
        1.0 / (best * 1e6)
    );
}

/// One continuous timing window around all `n` packed permutations, with no min over reps.
///
/// A multi-process benchmark wants the exact total work and the exact wall window for one
/// uninterrupted run. `mperm_per_s` is the same unit as `bench_packed` (millions of
/// packed-lane permutations per second), but taken from the one window instead of a minimum.
fn bench_packed_total(n: usize) {
    let perm = default_babybear_poseidon2_16();
    type P = <BabyBear as Field>::Packing;
    let lanes = P::WIDTH;
    let mut state: [P; 16] = std::array::from_fn(|i| P::from(BabyBear::new(i as u32 + 1)));
    for _ in 0..10_000 {
        perm.permute_mut(&mut state);
    }
    let t0 = Instant::now();
    for _ in 0..n {
        perm.permute_mut(&mut state);
    }
    let total = t0.elapsed();
    let secs = total.as_secs_f64();
    let units = n as f64 * lanes as f64;
    println!(
        "{{\"lanes\":{},\"units\":{},\"perms\":{},\"secs\":{},\"mperm_per_s\":{:.2}}}",
        lanes,
        units,
        n,
        secs,
        units / secs / 1e6
    );
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("bench") => {
            let n = std::env::args()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000);
            bench(n);
        }
        Some("bench-packed") => {
            let n = std::env::args()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000);
            bench_packed(n);
        }
        Some("bench-packed-total") => {
            let n = std::env::args()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000);
            bench_packed_total(n);
        }
        Some("kat") | None => kat(),
        Some(other) => {
            eprintln!("unknown mode {other:?}: use `kat`, `bench [n]`, `bench-packed [n]`, or `bench-packed-total [n]`");
            std::process::exit(2);
        }
    }
}
