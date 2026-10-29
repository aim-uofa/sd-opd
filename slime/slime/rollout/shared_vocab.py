"""SGLang generation restricted to the MiMo/Qwen3 shared action vocabulary."""

import json
from functools import lru_cache

from slime.utils.shared_vocab import MIMO_SHARED_VOCAB_SIZE, valid_student_prompt


class SharedVocabularyLogitProcessor:
    def __call__(self, logits, custom_param_list=None):
        if logits.shape[-1] < MIMO_SHARED_VOCAB_SIZE:
            raise ValueError("Rollout model vocabulary is smaller than the shared vocabulary")
        logits[..., MIMO_SHARED_VOCAB_SIZE:] = -float("inf")
        return logits


@lru_cache(maxsize=1)
def serialized_logit_processor():
    import dill

    # Keep this at the request top level: some routers discard logit_bias and
    # custom_params when reserializing sampling_params.
    return json.dumps({"callable": dill.dumps(SharedVocabularyLogitProcessor).hex()})


async def generate(args, sample, sampling_params, evaluation=False):
    from slime.rollout.sglang_rollout import GenerateState, generate as generate_sample

    if (
        args.opd_shared_vocab_size != MIMO_SHARED_VOCAB_SIZE or sample.multimodal_inputs
        or not getattr(args, "sglang_enable_custom_logit_processor", False)
    ):
        raise ValueError("Shared-vocabulary rollout requires text-only MiMo OPD and custom logit processing")
    if sampling_params.get("logit_bias") or sampling_params.get("custom_params"):
        raise ValueError("Shared-vocabulary rollout does not accept additional logit bias or custom params")
    state = GenerateState(args)
    tokens = sample.tokens or state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    if not valid_student_prompt(tokens):
        raise ValueError("Prompt contains unsupported tokens outside the shared vocabulary/Qwen3 thinking prefix")
    sample = await generate_sample(
        args, sample, sampling_params, custom_logit_processor=serialized_logit_processor()
    )
    response = sample.tokens[-sample.response_length:] if sample.response_length else []
    invalid = sorted({token for token in response if token < 0 or token >= MIMO_SHARED_VOCAB_SIZE})
    if invalid:
        raise ValueError(f"Rollout returned tokens outside the shared vocabulary: {invalid[:16]}")
    return sample
