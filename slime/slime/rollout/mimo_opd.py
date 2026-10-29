"""Collect MiMo teacher log-probs for the existing vanilla OPD advantage path."""

import math

import aiohttp


async def reward_func(args, sample, **kwargs):
    if sample.multimodal_inputs:
        raise ValueError("MiMo OPD supports text-only samples")
    payload = {"input_ids": sample.tokens, "response_length": sample.response_length}
    # The teacher serializes forwards; allow a complete rollout's scoring queue.
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.rm_url, json=payload) as response:
            response.raise_for_status()
            return await response.json()


def post_process_rewards(args, samples, **kwargs):
    for sample in samples:
        reward = sample.get_reward_value(args)
        values = reward.get("teacher_log_probs") if isinstance(reward, dict) else None
        if (
            not isinstance(values, list) or len(values) != sample.response_length
            or any(type(value) not in (int, float) or not math.isfinite(value) or value > 1e-5 for value in values)
        ):
            raise ValueError("MiMo scorer returned invalid or misaligned teacher log-probabilities")
        sample.teacher_log_probs = values
    rewards = [0.0] * len(samples)
    return rewards, rewards.copy()
