"""Collect student/base advantages with current-actor centering before optimizer updates."""

import hashlib
import io
import json
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.distributed as dist

from slime.utils.distillation_scoring import student_topk
from slime.utils.shared_vocab import restrict_logits
from slime.utils.student_base_delta import identity, signal_keys


def request_scores(args, tokens, student, support_ids, support_logps, vocab_size):
    payload = io.BytesIO()
    np.savez(payload, tokens=tokens.detach().cpu().numpy().astype(np.int64),
             student=student.detach().float().cpu().numpy(),
             support_ids=support_ids.detach().cpu().numpy().astype(np.int32),
             support_logps=support_logps.detach().float().cpu().numpy(), vocab_size=np.int64(vocab_size))
    body = payload.getvalue()
    expected_metadata = dict(identity(args.student_base_delta_mode, args.student_base_delta_strength,
                                      args.student_base_delta_top_k,
                                      getattr(args, "student_base_delta_student_source", "current"),
                                      getattr(args, "opd_shared_vocab_size", 0)),
                             vocab_size=vocab_size, request_sha256=hashlib.sha256(body).hexdigest())
    request = urllib.request.Request(args.student_base_delta_scorer_url, data=body,
                                     headers={"Content-Type": "application/octet-stream"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=args.student_base_delta_scorer_timeout) as response:
            result = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Student-base delta scorer rejected the batch: {error.read().decode(errors='replace')}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Student-base delta scorer unavailable: {error}") from error
    with np.load(io.BytesIO(result), allow_pickle=False) as arrays:
        keys = signal_keys(args.student_base_delta_mode)
        if set(arrays.files) != set(keys) | {"metadata"}:
            raise ValueError("Unexpected student-base delta response fields.")
        if json.loads(arrays["metadata"].tobytes()) != expected_metadata:
            raise ValueError("Scorer method/mode/strength/top-k/student-source or request identity mismatch.")
        signals = {key: torch.as_tensor(arrays[key].copy(), device=student.device, dtype=torch.float32)
                   for key in keys}
    if any(value.shape != student.shape or not torch.isfinite(value).all() for value in signals.values()):
        raise ValueError("Nonfinite or misaligned student-base delta scores.")
    return signals


@torch.no_grad()
def student_topk_tp(logits, tokens, k, group, chunk_size=128):
    """Full-vocabulary probabilities from TP shards without gathering a whole response."""
    if logits.ndim != 2 or tokens.shape != logits.shape[:1] or not len(tokens) or chunk_size < 1:
        raise ValueError("Expected nonempty response-aligned TP logits/tokens and a positive chunk size.")
    world_size = dist.get_world_size(group)
    if not 1 <= k <= logits.shape[-1] * world_size:
        raise ValueError("Student-base delta top-k exceeds the global vocabulary.")
    sampled, ids, logps = [], [], []
    for start in range(0, len(tokens), chunk_size):
        local = logits[start:start + chunk_size].contiguous()
        shards = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(shards, local, group=group)
        lp, support, support_lp = student_topk(torch.cat(shards, dim=-1), tokens[start:start + chunk_size], k)
        sampled.append(lp)
        ids.append(support)
        logps.append(support_lp)
    return torch.cat(sampled), torch.cat(ids), torch.cat(logps)


def request_scores_tp(args, tokens, student, ids, logps, vocab_size, group):
    """One scorer request per TP group; deliver scores or failure to every rank."""
    source = dist.get_global_rank(group, 0)
    error, signals = None, None
    if dist.get_rank(group) == 0:
        try:
            signals = request_scores(args, tokens, student, ids, logps, vocab_size)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device=student.device)
    dist.broadcast(failed, src=source, group=group)
    if failed.item():
        message = [error]
        dist.broadcast_object_list(message, src=source, group=group)
        raise RuntimeError(f"Student-base delta TP scorer failed: {message[0]}")
    keys = signal_keys(args.student_base_delta_mode)
    if signals is not None:
        values = torch.stack([signals[key] for key in keys])
    else:
        values = torch.empty((len(keys), len(student)), dtype=torch.float32, device=student.device)
    dist.broadcast(values, src=source, group=group)
    return dict(zip(keys, values.unbind(0), strict=True))


@torch.no_grad()
def get_student_base_delta_log_probs_and_entropy(logits, *, args, unconcat_tokens,
                                                 total_lengths, response_lengths, **kwargs):
    from .loss import get_log_probs_and_entropy, get_responses

    output, result = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=unconcat_tokens, total_lengths=total_lengths,
        response_lengths=response_lengths, **kwargs,
    )
    responses = get_responses(logits, args=args, unconcat_tokens=unconcat_tokens,
                              total_lengths=total_lengths, response_lengths=response_lengths)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    tp_group = None
    if tp_size > 1:
        from megatron.core import mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        if dist.get_world_size(tp_group) != tp_size:
            raise ValueError("Student-base delta TP group size differs from the launch configuration.")
    for i, (scores, tokens) in enumerate(responses):
        scores = restrict_logits(scores, args)
        if tp_size == 1:
            student, ids, logps = student_topk(scores, tokens, args.student_base_delta_top_k)
        else:
            student, ids, logps = student_topk_tp(scores, tokens, args.student_base_delta_top_k, tp_group)
        aligned = torch.tensor(int(torch.allclose(student, result["log_probs"][i], atol=2e-4, rtol=2e-4)),
                               dtype=torch.int32, device=scores.device)
        if tp_size > 1:
            dist.all_reduce(aligned, op=dist.ReduceOp.MIN, group=tp_group)
        if not aligned.item():
            raise ValueError("Student-base delta probabilities disagree with the PPO old-policy probabilities.")
        if tp_size == 1:
            signals = request_scores(args, unconcat_tokens[i], student, ids, logps, scores.shape[-1])
        else:
            signals = request_scores_tp(args, unconcat_tokens[i], student, ids, logps,
                                        scores.shape[-1] * tp_size, tp_group)
        for key, value in signals.items():
            result.setdefault(key, []).append(value)
    return output, result


def apply_student_base_delta_advantages(args, rollout_data, advantages, student_log_probs):
    signals = rollout_data.get("sbd_advantages")
    ungated = args.student_base_delta_mode == "ungated"
    teachers = rollout_data.get("teacher_log_probs")
    if signals is None or student_log_probs is None or (not ungated and teachers is None):
        raise ValueError("Student-base delta requires pre-update actor/base scores and teacher scores when gated.")
    if (not (len(signals) == len(student_log_probs) == len(advantages))
        or (not ungated and len(teachers) != len(signals))):
        raise ValueError("Student-base delta sample counts do not match.")
    if any(reward != 0 for reward in rollout_data.get("rewards", [])):
        raise ValueError("Student-base delta expects zero task rewards.")
    reverse_kls = []
    for i, (signal, student, old_adv) in enumerate(zip(signals, student_log_probs, advantages, strict=True)):
        if (not (signal.shape == student.shape == old_adv.shape)
            or (not ungated and teachers[i].shape != signal.shape)):
            raise ValueError("Student-base delta response-token shapes do not match.")
        if not torch.isfinite(signal).all():
            raise ValueError("Student-base delta advantages must be finite.")
        advantages[i] = (args.opd_kl_coef * signal.to(old_adv.device)).detach()
        if not ungated:
            reverse_kls.append((student - teachers[i].to(student.device)).detach())
    if not ungated:
        rollout_data["opd_reverse_kl"] = reverse_kls
