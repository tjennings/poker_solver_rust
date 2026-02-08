//! Benchmarks for tensor-based CFR operations.
//!
//! Compares CPU (ndarray) vs GPU (WGPU) performance for tensor operations.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use burn::backend::ndarray::NdArray;
use burn::backend::wgpu::Wgpu;
use burn::prelude::*;

use poker_solver_core::cfr::{
    accumulate_strategy, compute_average_strategy, regret_match_tensor, update_regrets_cfr_plus,
};

type CpuBackend = NdArray;
type GpuBackend = Wgpu;

/// Create test data for benchmarking.
fn create_test_data<B: Backend>(
    num_info_sets: usize,
    max_actions: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2, Bool>) {
    // Create random-ish regrets
    let regrets_data: Vec<f32> = (0..num_info_sets * max_actions)
        .map(|i| ((i % 7) as f32) - 3.0) // Range from -3 to 3
        .collect();

    let regrets = Tensor::<B, 1>::from_floats(regrets_data.as_slice(), device)
        .reshape([num_info_sets, max_actions]);

    // Create action mask (assume all actions valid for simplicity)
    let mask_data: Vec<i32> = vec![1i32; num_info_sets * max_actions];
    let action_mask = Tensor::<B, 1, Int>::from_ints(mask_data.as_slice(), device)
        .reshape([num_info_sets, max_actions])
        .equal_elem(1);

    (regrets, action_mask)
}

/// Benchmark regret matching on CPU.
fn bench_regret_match_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("regret_match_cpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (regrets, mask) = create_test_data::<CpuBackend>(size, 4, &device);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result =
                    regret_match_tensor(black_box(regrets.clone()), black_box(mask.clone()));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark regret matching on GPU.
fn bench_regret_match_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("regret_match_gpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (regrets, mask) = create_test_data::<GpuBackend>(size, 4, &device);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result =
                    regret_match_tensor(black_box(regrets.clone()), black_box(mask.clone()));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark CFR+ regret update on CPU.
fn bench_update_regrets_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_regrets_cpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (regrets, _) = create_test_data::<CpuBackend>(size, 4, &device);
        let delta = regrets.clone();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result =
                    update_regrets_cfr_plus(black_box(regrets.clone()), black_box(delta.clone()));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark CFR+ regret update on GPU.
fn bench_update_regrets_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_regrets_gpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (regrets, _) = create_test_data::<GpuBackend>(size, 4, &device);
        let delta = regrets.clone();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result =
                    update_regrets_cfr_plus(black_box(regrets.clone()), black_box(delta.clone()));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark strategy accumulation on CPU.
fn bench_accumulate_strategy_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulate_strategy_cpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (strategy_sum, _) = create_test_data::<CpuBackend>(size, 4, &device);
        let (strategy, _) = create_test_data::<CpuBackend>(size, 4, &device);
        let reach = Tensor::<CpuBackend, 1>::ones([size], &device);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result = accumulate_strategy(
                    black_box(strategy_sum.clone()),
                    black_box(strategy.clone()),
                    black_box(reach.clone()),
                );
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark strategy accumulation on GPU.
fn bench_accumulate_strategy_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulate_strategy_gpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (strategy_sum, _) = create_test_data::<GpuBackend>(size, 4, &device);
        let (strategy, _) = create_test_data::<GpuBackend>(size, 4, &device);
        let reach = Tensor::<GpuBackend, 1>::ones([size], &device);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result = accumulate_strategy(
                    black_box(strategy_sum.clone()),
                    black_box(strategy.clone()),
                    black_box(reach.clone()),
                );
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark average strategy computation on CPU.
fn bench_compute_average_strategy_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_average_strategy_cpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (strategy_sum, mask) = create_test_data::<CpuBackend>(size, 4, &device);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result = compute_average_strategy(
                    black_box(strategy_sum.clone()),
                    black_box(mask.clone()),
                );
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark average strategy computation on GPU.
fn bench_compute_average_strategy_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_average_strategy_gpu");

    for size in [100, 1_000, 10_000, 100_000] {
        let device = Default::default();
        let (strategy_sum, mask) = create_test_data::<GpuBackend>(size, 4, &device);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let result = compute_average_strategy(
                    black_box(strategy_sum.clone()),
                    black_box(mask.clone()),
                );
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_regret_match_cpu,
    bench_regret_match_gpu,
    bench_update_regrets_cpu,
    bench_update_regrets_gpu,
    bench_accumulate_strategy_cpu,
    bench_accumulate_strategy_gpu,
    bench_compute_average_strategy_cpu,
    bench_compute_average_strategy_gpu,
);

criterion_main!(benches);
