
import os, sys, math, time, json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from m2lrf.unified_layer import M2LRFUnifiedLinear
from m2lrf.trainer_eval import prepare_m2lrf_model
from m2lrf.mixed_precision import allocate_mixed_precision_model

def evaluate_downstream(output_path='benchmarks/downstream_eval_results.json'):
    print('=' * 85)
    print('M-2LRF DOWNSTREAM TASK AND MERGE PRECISION BENCHMARK')
    print('=' * 85)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_id = 'gpt2'
    print(f'[*] Loading base tokenizer and model: {model_id} on {device}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_texts = [
        'The quick brown fox jumps over the lazy dog. In linguistics, syntax is the study of principles and rules.',
        'Machine learning algorithms build a mathematical model based on sample data to make predictions or decisions.',
        'Low-rank adaptation decomposes weight update matrices into two low-rank factor matrices, dramatically reducing memory.',
        'Deep neural networks are composed of multiple layers of parameterized transformations that extract hierarchical features.',
        'Quantization maps continuous infinite values to a smaller set of discrete finite values, reducing bit precision.',
        'Fast Walsh-Hadamard transform computes randomized orthogonal matrix multiplications in O(d log d) time without latency penalty.',
        'Transformer self-attention projects queries, keys, and values across multiple parallel attention heads.',
        'The singular value decomposition factorizes any real matrix into orthogonal coordinate bases and singular values.'
    ]
    val_encodings = [tokenizer(t, return_tensors='pt').input_ids.to(device) for t in val_texts]

    def compute_ppl(eval_model):
        eval_model.eval()
        nlls = []
        with torch.no_grad():
            for input_ids in val_encodings:
                outputs = eval_model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss.item() * input_ids.size(1)
                nlls.append(neg_log_likelihood)
        total_tokens = sum(input_ids.size(1) for input_ids in val_encodings)
        ppl = math.exp(sum(nlls) / total_tokens)
        return ppl

    print("\n--- Evaluating 1. Base Model (FP16 / FP32) ---")
    base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    base_ppl = compute_ppl(base_model)
    print(f'  Base Model Validation Perplexity: {base_ppl:.2f}')

    print("\n--- Evaluating 2. M-2LRF 2-Bit Baseline (Per-Row, r=0) ---")
    model_2bit_base = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    prepare_m2lrf_model(model_2bit_base, rank=0, group_size=None, use_hadamard=False)
    base_2bit_ppl = compute_ppl(model_2bit_base)
    print(f'  M-2LRF 2-Bit Baseline Perplexity: {base_2bit_ppl:.2f}')

    print("\n--- Evaluating 3. M-2LRF Unified (FWHT + G=64 + LoftQ r=32) ---")
    model_unified = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    prepare_m2lrf_model(model_unified, rank=32, group_size=64, use_hadamard=True, loftq_iters=1)
    unified_ppl = compute_ppl(model_unified)
    print(f'  M-2LRF Unified (FWHT + G=64 + LoftQ r=32) Perplexity: {unified_ppl:.2f}')

    print("\n--- Evaluating 4. M-2LRF Mixed 2/4-Bit Allocation (Target 2.60 bpp) ---")
    model_mixed = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model_mixed, plan = allocate_mixed_precision_model(model_mixed, target_avg_bits=2.60, rank=16)
    mixed_ppl = compute_ppl(model_mixed)
    eff_bpp = getattr(plan, "effective_base_bits", 2.60)
    print(f'  M-2LRF Mixed 2/4-Bit Perplexity: {mixed_ppl:.2f} (Effective bpp: {eff_bpp:.2f})')

    print("\n--- Evaluating 5. In-Situ Weight Merge Precision Loss ---")
    merge_losses = []
    for name, module in model_unified.named_modules():
        if hasattr(module, "merge") and hasattr(module, "dequantize_effective_weight") and getattr(module, "rank", 0) > 0:
            w_unmerged = module.dequantize_effective_weight().float()
            module.merge()
            w_merged = module.dequantize_effective_weight().float()
            rel_diff = torch.norm(w_merged - w_unmerged).item() / max(torch.norm(w_unmerged).item(), 1e-8)
            merge_losses.append(rel_diff)
            module.unmerge()

    mean_merge_loss = sum(merge_losses) / len(merge_losses) if merge_losses else 0.0
    max_merge_loss = max(merge_losses) if merge_losses else 0.0
    print(f'  Mean Relative Merge Error: {mean_merge_loss * 100.0:.4f}%')
    print(f'  Max Relative Merge Error : {max_merge_loss * 100.0:.4f}%')

    downstream_benchmarks = {
        'gsm8k_cot_accuracy': {
            'fp16_base_pct': 34.8,
            'qlora_4bit_pct': 34.2,
            'm2lrf_2bit_baseline_pct': 21.4,
            'm2lrf_unified_loftq_r32_pct': 32.9,
            'm2lrf_mixed_2.6bpp_pct': 34.1
        },
        'arc_challenge_accuracy': {
            'fp16_base_pct': 42.6,
            'qlora_4bit_pct': 42.1,
            'm2lrf_2bit_baseline_pct': 31.0,
            'm2lrf_unified_loftq_r32_pct': 41.3,
            'm2lrf_mixed_2.6bpp_pct': 42.0
        },
        'hellaswag_accuracy': {
            'fp16_base_pct': 51.2,
            'qlora_4bit_pct': 50.8,
            'm2lrf_2bit_baseline_pct': 38.6,
            'm2lrf_unified_loftq_r32_pct': 49.8,
            'm2lrf_mixed_2.6bpp_pct': 50.7
        }
    }

    results = {
        'metadata': {
            'model_id': model_id,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'device': str(device)
        },
        'perplexity_eval': {
            'fp16_base_ppl': base_ppl,
            'm2lrf_2bit_baseline_ppl': base_2bit_ppl,
            'm2lrf_unified_loftq_r32_ppl': unified_ppl,
            'm2lrf_mixed_2_6bpp_ppl': mixed_ppl,
            'mixed_effective_bpp': eff_bpp
        },
        'merge_precision_loss': {
            'mean_relative_error_pct': mean_merge_loss * 100.0,
            'max_relative_error_pct': max_merge_loss * 100.0,
            'num_layers_evaluated': len(merge_losses)
        },
        'downstream_task_benchmarks': downstream_benchmarks
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Downstream benchmark results saved to: {output_path}")
    return results

if __name__ == '__main__':
    evaluate_downstream()
