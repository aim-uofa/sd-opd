"""Full-softmax scoring helpers for on-policy distillation."""

import torch


@torch.no_grad()
def student_topk(logits, tokens, k, chunk_size=128):
    """Return sampled log p, support IDs and log p on that support in FP32."""
    if logits.ndim != 2 or tokens.shape != logits.shape[:1] or not len(tokens):
        raise ValueError("Distillation needs nonempty, response-aligned logits and token IDs.")
    if not 1 <= k <= logits.shape[-1] or chunk_size < 1:
        raise ValueError("Distillation top-k must be in [1, vocab_size] and chunk_size must be positive.")
    sampled, ids, logps = [], [], []
    for start in range(0, len(tokens), chunk_size):
        scores = logits[start : start + chunk_size].float()
        chosen = tokens[start : start + chunk_size].long().unsqueeze(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("Distillation student logits must be finite.")
        normalizer = scores.logsumexp(-1, keepdim=True)
        support = scores.topk(k, dim=-1).indices
        missing = ~(support == chosen).any(-1)
        support[missing, -1] = chosen[missing, 0]
        sampled.append((scores.gather(-1, chosen) - normalizer).squeeze(-1))
        ids.append(support)
        logps.append(scores.gather(-1, support) - normalizer)
    return torch.cat(sampled), torch.cat(ids), torch.cat(logps)


@torch.inference_mode()
def score_model(model, tokens, response_length, support_ids, chunk_size, shared_vocab_size=0):
    """Score chosen IDs on the full-vocabulary softmax; preserve the causal shift."""
    if chunk_size < 1 or not 0 < response_length < len(tokens):
        raise ValueError("Distillation requires a nonempty prompt and response, and a positive chunk size.")
    if shared_vocab_size and not 0 < shared_vocab_size <= model.config.vocab_size:
        raise ValueError("Shared vocabulary exceeds the scoring model's vocabulary")
    device = next(model.parameters()).device
    input_ids = tokens[:-1].to(device=device, dtype=torch.long).unsqueeze(0)
    prompt_length = len(tokens) - response_length
    hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
    hidden = hidden[0, prompt_length - 1 :]
    sampled, support = [], []
    targets = tokens[-response_length:].to(device=device, dtype=torch.long)
    for start in range(0, response_length, chunk_size):
        stop = min(start + chunk_size, response_length)
        logits = model.lm_head(hidden[start:stop]).float()
        if shared_vocab_size:
            logits = logits[:, :shared_vocab_size]
        normalizer = logits.logsumexp(-1, keepdim=True)
        sampled.append((logits.gather(-1, targets[start:stop, None]) - normalizer).squeeze(-1).cpu())
        ids = support_ids[start:stop].to(device=device, dtype=torch.long)
        support.append((logits.gather(-1, ids) - normalizer).cpu())
    return torch.cat(sampled), torch.cat(support)
