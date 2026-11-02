"""Zero task rewards; current-student/base scoring runs before actor updates."""


async def reward_func(args, sample, **kwargs):
    return 0.0


def post_process_rewards(args, samples, **kwargs):
    if not samples or any(sample.get_reward_value(args) != 0.0 for sample in samples):
        raise ValueError("Student-base delta requires nonempty rollouts with zero task rewards.")
    if any((sample.metadata or {}).get("source_name") in
           ("keep_floor", "add_sft", "keep_failure_replay", "budget_shortcut") for sample in samples):
        raise ValueError("Student-base delta does not use replay samples.")
    rewards = [0.0] * len(samples)
    return rewards, rewards.copy()
