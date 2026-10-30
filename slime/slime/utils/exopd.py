"""ExOPD with a frozen initial-student reference."""

import math

import torch


def validate_exopd_args(args):
    """Keep the reference frozen and the vanilla OPD training path intact."""
    if not args.use_opd or getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("ExOPD requires --use-opd and the Megatron backend.")
    if not math.isfinite(args.opd_kl_coef) or args.opd_kl_coef <= 0:
        raise ValueError("ExOPD requires a finite, positive --opd-kl-coef.")
    if not args.ref_load or args.ref_update_interval is not None:
        raise ValueError("ExOPD requires a frozen initial-student --ref-load, without --ref-update-interval.")
    if (
        args.advantage_estimator != "grpo"
        or args.kl_coef != 0
        or args.use_kl_loss
        or args.normalize_advantages
        or args.entropy_coef != 0
        or args.custom_advantage_function_path
        or args.loss_type != "policy_loss"
        or getattr(args, "use_score_centering", False)
    ):
        raise ValueError("ExOPD uses zero-reward GRPO policy_loss without extra losses or advantage normalization.")
    if (
        args.use_rollout_logprobs
        or getattr(args, "keep_old_actor", False)
        or not args.compute_advantages_and_returns
    ):
        raise ValueError("ExOPD requires the pre-update current-actor log-probability and advantage pass.")
    module = "slime.rollout.mimo_opd" if args.opd_shared_vocab_size else "slime.rollout.on_policy_distillation"
    if (
        args.custom_rm_path != f"{module}.reward_func"
        or args.custom_reward_post_process_path != f"{module}.post_process_rewards"
        or args.reward_key
    ):
        raise ValueError(f"ExOPD requires the zero-reward hooks from {module}.")
    # External teacher scores are full response tensors; student/reference
    # scores would be local slices at CP > 1.
    if getattr(args, "context_parallel_size", 1) != 1:
        raise ValueError("ExOPD currently requires context parallel size 1.")


@torch.no_grad()
def apply_exopd_advantages(args, rollout_data, advantages, student_log_probs):
    """A = teacher - student + (lambda - 1) * (teacher - initial), in log-probs.

    Extrapolate the implicit reward only; scaling the entire vanilla advantage
    would incorrectly scale the current-student KL term as well.
    """
    teacher_log_probs = rollout_data.get("teacher_log_probs")
    reference_log_probs = rollout_data.get("ref_log_probs")
    if student_log_probs is None or teacher_log_probs is None or reference_log_probs is None:
        raise ValueError("ExOPD requires student, teacher_log_probs and frozen ref_log_probs.")
    if not (len(advantages) == len(student_log_probs) == len(teacher_log_probs) == len(reference_log_probs)):
        raise ValueError("ExOPD requires matching sample counts for advantages and all three policies.")
    if any(reward != 0 for reward in rollout_data.get("rewards", [])):
        raise ValueError("ExOPD expects zero task rewards.")

    reverse_kls = []
    for i, (adv, student, teacher, reference) in enumerate(
        zip(advantages, student_log_probs, teacher_log_probs, reference_log_probs, strict=True)
    ):
        teacher = teacher.to(device=student.device, dtype=torch.float32)
        reference = reference.to(device=student.device, dtype=torch.float32)
        student = student.float()
        if not (adv.shape == student.shape == teacher.shape == reference.shape):
            raise ValueError("ExOPD requires aligned response-token log-probabilities.")
        reverse_kl = student - teacher
        advantages[i] = adv + args.opd_kl_coef * (
            -reverse_kl + (args.exopd_scale - 1.0) * (teacher - reference)
        )
        reverse_kls.append(reverse_kl)
    # Preserve the existing teacher/student KL metric.
    rollout_data["opd_reverse_kl"] = reverse_kls
