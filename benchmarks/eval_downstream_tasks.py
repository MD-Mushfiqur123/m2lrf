
import sys, math, time, json

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
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from m2lrf.trainer_eval import prepare_m2lrf_model
from m2lrf.mixed_precision import allocate_mixed_precision_model


def load_wikitext2_validation_tokens(tokenizer):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "WikiText-2 evaluation requires the optional 'datasets' package."
        ) from exc

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    if tokens.size(1) < 2:
        raise ValueError("WikiText-2 validation split produced fewer than two tokens.")
    return tokens


def compute_perplexity(eval_model, tokens, stride=512):
    eval_model.eval()
    max_length = getattr(
        eval_model.config,
        "max_position_embeddings",
        getattr(eval_model.config, "n_positions", 1024),
    )
    max_length = int(max_length)
    stride = min(int(stride), max_length - 1)
    if max_length < 2 or stride < 1:
        raise ValueError("Model context length and evaluation stride must be positive.")

    device = next(eval_model.parameters()).device
    token_count = tokens.size(1)
    total_nll = 0.0
    scored_tokens = 0
    with torch.no_grad():
        for target_start in range(1, token_count, stride):
            target_end = min(target_start + stride, token_count)
            context_start = max(0, target_end - max_length)
            input_ids = tokens[:, context_start:target_end].to(device)
            outputs = eval_model(input_ids)

            first_logit = target_start - context_start - 1
            target_count = target_end - target_start
            logits = outputs.logits[:, first_logit:first_logit + target_count, :]
            targets = tokens[:, target_start:target_end].to(device)
            total_nll += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            ).item()
            scored_tokens += target_count

    return math.exp(total_nll / scored_tokens), scored_tokens


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

    validation_tokens = load_wikitext2_validation_tokens(tokenizer)

    print("\n--- Evaluating 1. Base Model ---")
    base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    base_dtype = str(next(base_model.parameters()).dtype)
    print(f"  Base model dtype: {base_dtype}")
    base_ppl, scored_tokens = compute_perplexity(base_model, validation_tokens)
    print(f'  Base Model Validation Perplexity: {base_ppl:.2f}')

    print("\n--- Evaluating 2. M-2LRF 2-Bit Baseline (Per-Row, r=0) ---")
    model_2bit_base = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    prepare_m2lrf_model(model_2bit_base, rank=0, group_size=None, use_hadamard=False)
    base_2bit_ppl, _ = compute_perplexity(model_2bit_base, validation_tokens)
    print(f'  M-2LRF 2-Bit Baseline Perplexity: {base_2bit_ppl:.2f}')

    print("\n--- Evaluating 3. M-2LRF Unified (FWHT + G=64 + LoftQ r=32) ---")
    model_unified = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    prepare_m2lrf_model(model_unified, rank=32, group_size=64, use_hadamard=True, loftq_iters=1)
    unified_ppl, _ = compute_perplexity(model_unified, validation_tokens)
    print(f'  M-2LRF Unified (FWHT + G=64 + LoftQ r=32) Perplexity: {unified_ppl:.2f}')

    print("\n--- Evaluating 4. M-2LRF Mixed 2/4-Bit Allocation (Target 2.60 bpp) ---")
    model_mixed = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model_mixed, plan = allocate_mixed_precision_model(model_mixed, target_avg_bits=2.60, rank=16)
    mixed_ppl, _ = compute_perplexity(model_mixed, validation_tokens)
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

    results = {
        'metadata': {
            'model_id': model_id,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'device': str(device),
            'evaluation_dataset': 'Salesforce/wikitext wikitext-2-raw-v1 validation',
            'scored_tokens': scored_tokens,
            'model_dtype': base_dtype,
            'note': 'Evaluated on GPT-2 (124M). High-level reasoning benchmarks (GSM8K, ARC, MMLU) require instruction-tuned 7B+ models and are reserved for future cluster runs.'
        },
        'perplexity_eval': {
            'base_model_ppl': base_ppl,
            'm2lrf_2bit_baseline_ppl': base_2bit_ppl,
            'm2lrf_unified_loftq_r32_ppl': unified_ppl,
            'm2lrf_mixed_2_6bpp_ppl': mixed_ppl,
            'mixed_effective_bpp': eff_bpp,
            'perplexity_reduction_ratio': base_2bit_ppl / max(unified_ppl, 1e-6)
        },
        'merge_precision_loss': {
            'mean_relative_error_pct': mean_merge_loss * 100.0,
            'max_relative_error_pct': max_merge_loss * 100.0,
            'num_layers_evaluated': len(merge_losses)
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Downstream benchmark results saved to: {output_path}")
    return results

if __name__ == '__main__':
    evaluate_downstream()
