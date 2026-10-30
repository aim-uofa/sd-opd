"""Shared action vocabulary and model-specific prompts for MiMo/Qwen3 OPD."""

import json
import math
from pathlib import Path

MIMO_SHARED_VOCAB_SIZE = 151665
QWEN3_THINK_TOKEN_IDS = (151667, 151668)
SHARED_PROMPT_VERSION = 3
GENERATE_FUNCTION = "slime.rollout.shared_vocab.generate"
REWARD_MODULE = "slime.rollout.mimo_opd"

# Keep Qwen3's empty thinking block; /no_think alone is insufficient when
# <think> is excluded from the generated action vocabulary.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] }}"
    "{% if message['role'] == 'user' %}{{ '\n/no_think' }}{% endif %}"
    "{{ '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}"
)
CHAT_TEMPLATE_KWARGS = dict(chat_template=CHAT_TEMPLATE, enable_thinking=False)


def valid_student_prompt(tokens):
    return bool(len(tokens)) and all(
        0 <= token < MIMO_SHARED_VOCAB_SIZE or token in QWEN3_THINK_TOKEN_IDS for token in tokens
    )


class MiMoPromptAdapter:
    """Retokenize the entire prompt, preserving response IDs and BPE boundaries."""

    def __init__(self, student_tokenizer, teacher_tokenizer):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        vocab = student_tokenizer.get_vocab()
        if tuple(vocab.get(token) for token in ("<think>", "</think>")) != QWEN3_THINK_TOKEN_IDS:
            raise ValueError("Unexpected Qwen3 thinking token IDs")

    def __call__(self, prompt_ids):
        if not valid_student_prompt(prompt_ids):
            raise ValueError("Unsupported student prompt token IDs")
        text = self.student_tokenizer.decode(
            prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        ids = self.teacher_tokenizer.encode(text, add_special_tokens=False)
        if not ids or any(token < 0 or token >= MIMO_SHARED_VOCAB_SIZE for token in ids):
            raise ValueError("Retokenized MiMo prompt contains unsupported token IDs")
        return ids


def token_map(path):
    path = Path(path)
    raw = json.loads((path / "tokenizer.json").read_text())
    config = json.loads((path / "tokenizer_config.json").read_text())
    vocab = dict(raw["model"]["vocab"])
    for token in raw.get("added_tokens", []):
        vocab[token["content"]] = token["id"]
    for index, token in config.get("added_tokens_decoder", {}).items():
        if token["content"] in vocab and vocab[token["content"]] != int(index):
            raise ValueError(f"Conflicting added token ID in {path}")
        vocab[token["content"]] = int(index)
    return vocab


def validate_tokenizers(paths):
    """Check shared IDs, BPE segmentation and special-token behavior."""
    size = MIMO_SHARED_VOCAB_SIZE
    reference = None
    for path in map(Path, paths):
        mapping = {token: index for token, index in token_map(path).items() if index < size}
        if len(mapping) != size or set(mapping.values()) != set(range(size)):
            raise ValueError(f"Shared token-to-ID mapping is incomplete: {path}")
        raw = json.loads((path / "tokenizer.json").read_text())
        config = json.loads((path / "tokenizer_config.json").read_text())
        added = {
            item["id"]: {key: value for key, value in item.items() if key != "id"}
            for item in raw.get("added_tokens", []) if item["id"] < size
        }
        for index, item in config.get("added_tokens_decoder", {}).items():
            if int(index) < size:
                added[int(index)] = item
        model = dict(raw["model"])
        model["merges"] = [
            merge.split(" ") if isinstance(merge, str) else merge for merge in model.get("merges", [])
        ]
        model["ignore_merges"] = model.get("ignore_merges", False)
        signature = dict(
            mapping=mapping, model=model, added=added,
            **{key: raw.get(key) for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder")},
        )
        if reference is not None and signature != reference:
            raise ValueError(f"Shared tokenizer mapping/BPE pipeline differs: {path}")
        reference = signature


def restrict_logits(logits, args):
    size = getattr(args, "opd_shared_vocab_size", 0)
    if not size:
        return logits
    if getattr(args, "tensor_model_parallel_size", 1) != 1 or not 0 < size <= logits.shape[-1]:
        raise ValueError("Shared-vocabulary OPD requires TP=1 and a valid vocabulary prefix")
    return logits[..., :size]


def validate_args(args):
    if not args.use_opd or args.opd_type not in ("mimo", "exopd") or args.opd_shared_vocab_size != MIMO_SHARED_VOCAB_SIZE:
        raise ValueError("MiMo OPD requires --use-opd, --opd-type mimo or exopd, and --opd-shared-vocab-size 151665")
    if getattr(args, "train_backend", "megatron") != "megatron" or any(
        getattr(args, key, 1) != 1
        for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
    ):
        raise ValueError("MiMo OPD requires the Megatron backend with TP=PP=CP=1")
    if (
        args.custom_generate_function_path != GENERATE_FUNCTION
        or not getattr(args, "sglang_enable_custom_logit_processor", False)
        or not args.apply_chat_template
        or args.apply_chat_template_kwargs != CHAT_TEMPLATE_KWARGS
    ):
        raise ValueError("MiMo OPD requires masked generation and the shared ChatML template")
    if (
        args.rollout_temperature != 1 or args.rollout_top_p != 1 or args.rollout_top_k != -1
        or args.use_rollout_logprobs or args.partial_rollout
        or getattr(args, "update_weights_interval", 1) != 1
        or getattr(args, "keep_old_actor", False)
    ):
        raise ValueError("MiMo OPD requires fresh complete rollouts, actor log-probs, T=top-p=1 and top-k=-1")
    if (
        args.advantage_estimator != "grpo" or args.normalize_advantages
        or args.kl_coef != 0 or args.use_kl_loss or args.entropy_coef != 0
        or args.loss_type != "policy_loss" or not args.calculate_per_token_loss
        or getattr(args, "pg_loss_type", None) not in (None, "ppo")
        or getattr(args, "use_score_centering", False)
        or getattr(args, "use_tis", False) or getattr(args, "use_opsm", False)
        or getattr(args, "custom_advantage_function_path", None)
        or getattr(args, "custom_loss_function_path", None)
        or getattr(args, "custom_pg_loss_reducer_function_path", None)
    ):
        raise ValueError("MiMo OPD requires token-mean PPO with zero task reward and no extra losses")
    if (
        args.custom_rm_path != REWARD_MODULE + ".reward_func"
        or args.custom_reward_post_process_path != REWARD_MODULE + ".post_process_rewards"
        or args.reward_key is not None or not args.rm_url
        or not math.isfinite(args.opd_kl_coef) or args.opd_kl_coef <= 0
    ):
        raise ValueError("MiMo OPD requires its teacher-scoring hooks, scorer URL and a positive OPD coefficient")


if __name__ == "__main__":
    print(json.dumps(CHAT_TEMPLATE_KWARGS))
