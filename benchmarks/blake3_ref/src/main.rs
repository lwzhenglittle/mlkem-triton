// BLAKE3 vectors and the CPU rival, from the official Rust implementation.
//
//   cargo run --release -- kat   > ../../../data/vectors/blake3/kat.json
//   cargo run --release -- bench [bytes] [reps]
//
// The KAT inputs are deterministic (a repeating byte pattern), so the JSON is enough to check
// the numpy and Triton paths without shipping the input data.
use blake3;
use std::time::Instant;

fn fill(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn kat() {
    let lens = [
        0usize, 1, 2, 3, 4, 5, 63, 64, 65, 127, 128, 129, 511, 512, 1023, 1024, 1025, 2048, 2049,
        3072, 4096, 5000, 8192, 16384, 31744, 65536,
    ];
    println!("{{\"algorithm\":\"BLAKE3 (official Rust implementation, unkeyed)\",\"vectors\":[");
    for (i, &n) in lens.iter().enumerate() {
        let data = fill(n);
        let mut out = [0u8; 32];
        out.copy_from_slice(blake3::hash(&data).as_bytes());
        let hex: String = out.iter().map(|b| format!("{:02x}", b)).collect();
        if i > 0 {
            println!(",");
        }
        print!("{{\"len\":{},\"hash\":\"{}\"}}", n, hex);
    }
    println!("]}}");
}

fn bench(nbytes: usize, reps: usize) {
    let data = fill(nbytes);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let h = blake3::hash(&data);
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(h);
        if dt < best {
            best = dt;
        }
    }
    println!("{{\"bytes\":{},\"secs\":{:.9},\"gbs\":{:.4}}}", nbytes, best, nbytes as f64 / best / 1e9);
}

/// One continuous timing window around all `reps` hashes, with no per-rep minimum.
///
/// A multi-process benchmark wants the exact total work and the exact wall window for one
/// uninterrupted run: min-of-reps would under-report the elapsed time and inflate the rate.
/// The window starts after the buffer is allocated, so spawn/setup time stays outside it.
fn bench_total(nbytes: usize, reps: usize) {
    let data = fill(nbytes);
    let t0 = Instant::now();
    for _ in 0..reps {
        let h = blake3::hash(&data);
        std::hint::black_box(h);
    }
    let total = t0.elapsed();
    let secs = total.as_secs_f64();
    let units = reps as f64 * nbytes as f64;
    println!(
        "{{\"bytes_per_hash\":{},\"hashes\":{},\"units\":{},\"secs\":{},\"gbs\":{:.4}}}",
        nbytes,
        reps,
        units,
        secs,
        units / secs / 1e9
    );
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "kat".into());
    match mode.as_str() {
        "kat" => kat(),
        "bench" => {
            let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
            let reps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);
            bench(n, reps);
        }
        "bench-total" => {
            let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
            let reps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);
            bench_total(n, reps);
        }
        other => eprintln!(
            "unknown mode {other:?}: use `kat`, `bench [bytes] [reps]`, or `bench-total [bytes] [reps]`"
        ),
    }
}
