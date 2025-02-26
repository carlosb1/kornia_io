use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kornia_rs::read_image;
use std::path::PathBuf;

mod perf;
fn criterion_benchmark(c: &mut Criterion) {
    let PATH: PathBuf = [env!("CARGO_MANIFEST_DIR"), "clients", "test.jpg"]
        .iter()
        .collect();
    let str_path = PATH.into_os_string().into_string().unwrap();

    c.bench_function("benchmark", |b| {
        b.iter(|| {
            let info = read_image(str_path.clone());
        })
    });
}
/*
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
*/
criterion_group! {name = benches;
    config = Criterion::default().with_profiler(perf::FlamegraphProfiler::new(100));
    targets = criterion_benchmark
}
criterion_main!(benches);
