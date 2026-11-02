"""Current-actor-centered student/base deltas, optionally gated by teacher feedback."""

import math

import torch

METHOD = "student_base_delta"
VERSION = 1
SIGNAL_KEYS = (
    "teacher_log_probs", "student_base_log_probs", "sbd_teacher_advantages",
    "sbd_student_advantages", "sbd_advantages", "sbd_active", "sbd_topk_mass",
    "sbd_bonus", "sbd_saturated", "sbd_scale",
)
UNGATED_SIGNAL_KEYS = (
    "student_base_log_probs", "sbd_student_advantages", "sbd_advantages",
    "sbd_active", "sbd_topk_mass", "sbd_scale",
)


def signal_keys(mode):
    if mode == "vanilla":
        return ("teacher_log_probs", "sbd_advantages")
    return UNGATED_SIGNAL_KEYS if mode == "ungated" else SIGNAL_KEYS


def validate_method(mode, strength):
    if mode not in ("direct", "boost", "ungated", "vanilla"):
        raise ValueError("Student-base delta mode must be direct, boost, ungated or vanilla.")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Student-base delta strength must be finite and nonnegative.")
    if mode in ("direct", "ungated", "vanilla") and strength != 0:
        raise ValueError("Direct/ungated/vanilla mode has no strength parameter; set it to zero.")


def identity(mode, strength, top_k, student_source="current", shared_vocab_size=0):
    validate_method(mode, strength)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("Student-base delta top-k must be a positive integer.")
    if student_source not in ("current", "initial"):
        raise ValueError("Student-base delta student source must be current or initial.")
    result = dict(method=METHOD, version=VERSION, mode=mode, strength=strength, top_k=top_k,
                  student_source=student_source)
    if shared_vocab_size:
        from slime.utils.shared_vocab import SHARED_PROMPT_VERSION

        result["shared_vocab_size"] = shared_vocab_size
        result["shared_prompt_version"] = SHARED_PROMPT_VERSION
    return result


@torch.no_grad()
def delta_signals(student, teacher, base, student_support, teacher_support, base_support,
                  *, mode="boost", strength=0.25, initial_student=None, initial_student_support=None):
    """Use current-actor full-softmax weights, including for a frozen initial/base ratio."""
    validate_method(mode, strength)
    if mode == "vanilla":
        if (initial_student is not None or initial_student_support is not None
            or base is not None or base_support is not None or teacher is None
            or student.ndim != 1 or not student.numel() or teacher.shape != student.shape
            or any(not x.is_floating_point() or not torch.isfinite(x).all() or (x > 1e-4).any()
                   for x in (student, teacher))):
            raise ValueError("Vanilla OPD requires aligned finite student/teacher scores and no base")
        # Sampled reverse-KL OPD: no top-k centering and no sign gate.
        return dict(teacher_log_probs=teacher.detach(), sbd_advantages=(teacher - student).detach())
    if (initial_student is None) != (initial_student_support is None):
        raise ValueError("Provide both sampled and support scores for the initial student.")
    ratio_student = student if initial_student is None else initial_student
    ratio_support = student_support if initial_student_support is None else initial_student_support
    if (student.ndim != 1 or not student.numel()
        or base.shape != student.shape
        or student_support.ndim != 2 or student_support.shape[0] != len(student)
        or not student_support.shape[1]
        or base_support.shape != student_support.shape
        or ratio_student.shape != student.shape or ratio_support.shape != student_support.shape):
        raise ValueError("Expected aligned sampled [response] and support [response, k] log-probabilities.")
    tensors = (student, base, student_support, base_support, ratio_student, ratio_support)
    if mode == "ungated":
        if teacher is not None or teacher_support is not None:
            raise ValueError("Ungated mode does not use teacher scores.")
    else:
        if (teacher is None or teacher_support is None or teacher.shape != student.shape
            or teacher_support.shape != student_support.shape):
            raise ValueError("Expected aligned teacher log-probabilities.")
        tensors += (teacher, teacher_support)
    if any(not x.is_floating_point() or x.device != student.device
           or not torch.isfinite(x).all() or (x > 1e-4).any() for x in tensors):
        raise ValueError("Expected finite log-probabilities on one device.")
    # Keep FP64 for numerical verification, and promote low-precision scores.
    dtype = torch.float64 if all(x.dtype == torch.float64 for x in tensors) else torch.float32
    converted = [x.detach().to(dtype) for x in tensors]
    student, base, student_support, base_support, ratio_student, ratio_support = converted[:6]
    weights = student_support.exp()
    mass = weights.sum(-1)
    if (mass > 1.0001).any():
        raise ValueError("Student support probability mass exceeds one.")
    a_student = ratio_student - base - (weights * (ratio_support - base_support)).sum(-1)
    if mode == "ungated":
        if not torch.isfinite(a_student).all():
            raise ValueError("Nonfinite student-base delta signal; no fallback was applied.")
        return dict(student_base_log_probs=base, sbd_student_advantages=a_student,
                    sbd_advantages=a_student, sbd_active=torch.ones_like(a_student),
                    sbd_topk_mass=mass, sbd_scale=torch.ones_like(a_student))
    teacher, teacher_support = converted[6:]
    a_teacher = teacher - student - (weights * (teacher_support - student_support)).sum(-1)
    active = ((a_teacher > 0) & (a_student > 0)) | ((a_teacher < 0) & (a_student < 0))
    bonus = torch.where(active, a_teacher.sign() * torch.minimum(a_student.abs(), a_teacher.abs()), 0.0)
    advantages = (torch.where(active, a_student, 0.0) if mode == "direct"
                  else a_teacher + strength * bonus)
    # Bounded dimensionless scale for boost; zero means an absent teacher direction.
    scale = torch.zeros_like(advantages)
    nonzero = a_teacher != 0
    if mode == "boost":
        scale[nonzero] = 1 + strength * bonus[nonzero] / a_teacher[nonzero]
    else:
        # Direct mode has no bounded amplification factor; report coverage instead.
        scale = active.to(dtype)
    result = dict(
        teacher_log_probs=teacher, student_base_log_probs=base,
        sbd_teacher_advantages=a_teacher, sbd_student_advantages=a_student,
        sbd_advantages=advantages, sbd_active=active.to(dtype), sbd_topk_mass=mass,
        sbd_bonus=bonus if mode == "boost" else torch.zeros_like(bonus),
        sbd_saturated=(active & (a_student.abs() >= a_teacher.abs())).to(dtype), sbd_scale=scale,
    )
    if any(not torch.isfinite(x).all() for x in result.values()):
        raise ValueError("Nonfinite student-base delta signal; no fallback was applied.")
    return result


def validate_args(args):
    if (not args.use_opd or args.opd_type != METHOD
        or getattr(args, "train_backend", "megatron") != "megatron"):
        raise ValueError("Student-base delta requires --use-opd --opd-type student_base_delta and Megatron.")
    validate_scorer_args(args)


def validate_scorer_args(args):
    """Validate the scorer/PPO settings for student-base delta distillation."""
    identity(args.student_base_delta_mode, args.student_base_delta_strength, args.student_base_delta_top_k,
             getattr(args, "student_base_delta_student_source", "current"))
    if getattr(args, "opd_shared_vocab_size", 0):
        from slime.utils.shared_vocab import validate_args as validate_shared_args

        validate_shared_args(args)
    if args.student_base_delta_mode == "vanilla" and (
        not getattr(args, "opd_shared_vocab_size", 0) or args.student_base_delta_top_k != 1
    ):
        raise ValueError("Vanilla scorer mode requires the MiMo shared-vocabulary entry and top-k=1")
    if not args.student_base_delta_scorer_url:
        raise ValueError("Student-base delta requires its scorer URL.")
    if not math.isfinite(args.student_base_delta_scorer_timeout) or args.student_base_delta_scorer_timeout <= 0:
        raise ValueError("Scorer timeout must be finite and positive.")
    if not math.isfinite(args.opd_kl_coef) or args.opd_kl_coef <= 0:
        raise ValueError("OPD coefficient must be finite and positive.")
    if getattr(args, "tensor_model_parallel_size", 1) not in (1, 2):
        raise ValueError("Student-base delta supports TP=1 or TP=2.")
    for key in ("pipeline_model_parallel_size", "context_parallel_size"):
        if getattr(args, key, 1) != 1:
            raise ValueError("Student-base delta currently requires PP=CP=1; tensor/data parallelism is supported.")
    if (getattr(args, "update_weights_interval", 1) != 1 or getattr(args, "partial_rollout", False)
        or not getattr(args, "compute_advantages_and_returns", True)
        or args.rollout_temperature != 1 or args.rollout_top_p != 1
        or getattr(args, "rollout_top_k", -1) != -1 or args.use_rollout_logprobs
        or getattr(args, "keep_old_actor", False)):
        raise ValueError("Student-base delta requires fresh complete rollouts, T=top-p=1, top-k=-1 and actor old-policy scores.")
    if (args.advantage_estimator != "grpo" or args.normalize_advantages or args.kl_coef != 0
        or args.use_kl_loss or getattr(args, "entropy_coef", 0) != 0
        or getattr(args, "custom_advantage_function_path", None)
        or getattr(args, "custom_loss_function_path", None)
        or getattr(args, "custom_pg_loss_reducer_function_path", None)
        or getattr(args, "loss_type", "policy_loss") != "policy_loss"
        or not getattr(args, "calculate_per_token_loss", False)):
        raise ValueError("Student-base delta requires zero-reward GRPO, token-mean policy_loss and no extra losses.")
    flags = ("use_keep_floor", "use_keep_failure_replay", "use_budget_shortcut", "use_rcd",
             "opd_capability_mixing", "opd_correctness_gating", "opd_completion_aware_gating",
             "use_tis", "use_opsm", "use_critic")
    if any(getattr(args, key, False) for key in flags) or getattr(args, "add_sft_samples_per_rollout", 0):
        raise ValueError("Student-base delta cannot mix replay, correctness gating, other methods or importance corrections.")
    module = "slime.rollout.student_base_delta_distillation"
    if (getattr(args, "custom_rm_path", None) != module + ".reward_func"
        or getattr(args, "custom_reward_post_process_path", None) != module + ".post_process_rewards"
        or getattr(args, "reward_key", None)):
        raise ValueError("Student-base delta requires its own zero-task-reward hooks.")
