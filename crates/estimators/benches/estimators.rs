//! Criterion micro-benchmarks for the estimators at a realistic geo-panel size
//! (200 units × 130 periods). Run with `cargo bench -p panelkit-estimators`.

use criterion::{criterion_group, criterion_main, Criterion};
use panelkit_estimators::mcnnm::{fit_mcnnm_at, McnnmConfig};
use panelkit_estimators::sc::{fit_asc_at, fit_at, fit_sdid_at, AscConfig, ScConfig, SdidConfig};
use panelkit_estimators::Panel;
use panelkit_linalg::rng::Xoshiro256pp;
use panelkit_linalg::Mat;

fn make_panel(n: usize, t: usize, t0: usize) -> Panel {
    let mut rng = Xoshiro256pp::seed_from_u64(42);
    let rank = 3;
    let uf: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..rank).map(|_| rng.next_normal()).collect())
        .collect();
    let tf: Vec<Vec<f64>> = (0..t)
        .map(|_| (0..rank).map(|_| 0.5 * rng.next_normal()).collect())
        .collect();
    let ul: Vec<f64> = (0..n).map(|_| 10.0 + rng.next_normal()).collect();
    let mut tl = vec![0.0; t];
    let mut acc = 0.0;
    for v in tl.iter_mut() {
        acc += 0.02 * rng.next_normal();
        *v = acc;
    }
    let mut y = Mat::zeros(n, t);
    for i in 0..n {
        for p in 0..t {
            let mut v = ul[i] + tl[p];
            for k in 0..rank {
                v += uf[i][k] * tf[p][k];
            }
            if i == 0 && p >= t0 {
                v += 0.05;
            }
            y.set(i, p, v);
        }
    }
    Panel::block(y, &[0], t0)
}

fn bench(c: &mut Criterion) {
    let (n, t, t0) = (200usize, 130usize, 104usize);
    let panel = make_panel(n, t, t0);

    let mut g = c.benchmark_group("estimators_200x130");
    g.bench_function("sc", |b| {
        b.iter(|| fit_at(&panel, t0, ScConfig::default()))
    });
    g.bench_function("asc", |b| {
        b.iter(|| fit_asc_at(&panel, t0, AscConfig::default()))
    });
    g.bench_function("sdid", |b| {
        b.iter(|| fit_sdid_at(&panel, t0, SdidConfig::default()))
    });
    g.bench_function("mcnnm_fixed_lambda", |b| {
        let cfg = McnnmConfig {
            lambda: Some(1.0),
            ..Default::default()
        };
        b.iter(|| fit_mcnnm_at(&panel, t0, cfg))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
