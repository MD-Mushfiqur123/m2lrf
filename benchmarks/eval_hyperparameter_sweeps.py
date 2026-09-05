import os, sys, math, time, json
import torch
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from m2lrf.unified_layer import M2LRFUnifiedLinear
from m2lrf.quantizer import DualBasisQuantizer
from m2lrf.hadamard_transform import generate_synthetic_heavy_tailed_weights, calculate_kurtosis

def run_sweeps(output_path='benchmarks/hyperparameter_sweeps.json'):
    print('=' * 80)
    print('M-2LRF HYPERPARAMETER SWEEPS AND PARETO FRONTIER ANALYSIS')
    print('=' * 80)

    torch.manual_seed(42)
    in_f, out_f = 2048, 2048
    w = generate_synthetic_heavy_tailed_weights(out_f, in_f, num_outlier_channels=16, outlier_multiplier=12.0, seed=42)
    k0 = float(calculate_kurtosis(w))
    print(f'[*] Benchmark Matrix: {out_f}x{in_f} | Excess Kurtosis kappa_0 = {k0:.2f}')

    results = {
        'metadata': {
            'matrix_shape': [out_f, in_f],
            'pre_kurtosis': k0,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        },
        'fwht_block_size_sweep': [],
        'outlier_sigma_sweep': [],
        'lora_rank_sweep': []
    }

    print('\n--- 1. FWHT Block Size Sweep [64, 128, 256, 512, 1024] ---')
    for b in [64, 128, 256, 512, 1024]:
        layer = M2LRFUnifiedLinear(in_f, out_f, bits=2, group_size=64, use_hadamard=True, block_size=b, rank=0)
        layer.initialize_from_pretrained(w)
        w_eff = layer.dequantize_effective_weight().float()
        sqnr = DualBasisQuantizer.calculate_sqnr(w.float(), w_eff)

        x = torch.randn(2, 32, in_f)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = layer(x)
        lat = (time.perf_counter() - t0) / 20.0 * 1000.0

        mem = layer.memory_bytes() / (1024 ** 2)
        print(f'  Block Size B={b:4d} | SQNR: {sqnr:5.2f} dB | Latency: {lat:6.3f} ms | Footprint: {mem:.3f} MB')
        results['fwht_block_size_sweep'].append({
            'block_size': b, 'sqnr_db': sqnr, 'latency_ms': lat, 'memory_mb': mem
        })

    print('\n--- 2. Outlier Sigma Sweep [3.0, 3.5, 4.0, 4.5] ---')
    mean_val = w.mean().item()
    std_val = w.std().item()
    total_elements = out_f * in_f
    for sigma in [3.0, 3.5, 4.0, 4.5]:
        thresh = sigma * std_val
        outlier_mask = (w - mean_val).abs() > thresh
        num_outliers = outlier_mask.sum().item()
        density = (num_outliers / total_elements) * 100.0

        w_base = w.clone()
        sparse_vals = w_base[outlier_mask]
        w_base[outlier_mask] = 0.0

        layer = M2LRFUnifiedLinear(in_f, out_f, bits=2, group_size=64, use_hadamard=True, rank=0)
        layer.initialize_from_pretrained(w_base)
        w_eff = layer.dequantize_effective_weight().float()
        w_eff[outlier_mask] = sparse_vals
        sqnr = DualBasisQuantizer.calculate_sqnr(w.float(), w_eff)

        extra_bytes = num_outliers * (2 * 4 + 2)
        total_mem = (layer.memory_bytes() + extra_bytes) / (1024 ** 2)
        print(f'  Sigma={sigma:.1f} | Outliers: {num_outliers:5d} ({density:.3f} pct) | SQNR: {sqnr:.2f} dB | Mem: {total_mem:.3f} MB')
        results['outlier_sigma_sweep'].append({
            'sigma': sigma, 'num_outliers': num_outliers, 'outlier_density_pct': density, 'sqnr_db': sqnr, 'memory_mb': total_mem
        })

    print('\n--- 3. LoRA Rank Sweep [4, 8, 16, 32, 64] ---')
    for r in [4, 8, 16, 32, 64]:
        layer = M2LRFUnifiedLinear(in_f, out_f, bits=2, group_size=64, use_hadamard=True, rank=r, alpha=16.0)
        layer.initialize_from_pretrained(w, loftq_iters=1)
        w_eff = layer.dequantize_effective_weight().float()
        sqnr = DualBasisQuantizer.calculate_sqnr(w.float(), w_eff)
        trainable_params = layer.lora_A.numel() + layer.lora_B.numel()
        param_pct = (trainable_params / total_elements) * 100.0
        mem = layer.memory_bytes() / (1024 ** 2)
        print(f'  Rank r={r:2d} | Trainable Params: {trainable_params:6d} ({param_pct:.2f} pct) | Step-0 SQNR: {sqnr:.2f} dB | Mem: {mem:.3f} MB')
        results['lora_rank_sweep'].append({
            'rank': r, 'trainable_params': trainable_params, 'param_fraction_pct': param_pct, 'step0_sqnr_db': sqnr, 'memory_mb': mem
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\n[+] Sweeps saved to: {output_path}')
    return results

if __name__ == '__main__':
    run_sweeps()
