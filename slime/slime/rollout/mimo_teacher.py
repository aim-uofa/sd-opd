"""Frozen local MiMo teacher service for shared-vocabulary vanilla OPD."""

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path

import torch
from aiohttp import web
from transformers import AutoModelForCausalLM, AutoTokenizer

from slime.utils.shared_vocab import (
    MIMO_SHARED_VOCAB_SIZE,
    SHARED_PROMPT_VERSION,
    MiMoPromptAdapter,
    token_map,
    validate_tokenizers,
)

logger = logging.getLogger(__name__)


def model_metadata(path, role, max_context):
    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    keys = (
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
    )
    expected = {
        "student": ("qwen3", (2048, 6144, 28, 16, 8, 151936)),
        "teacher": ("mimo", (4096, 11008, 36, 32, 8, 151680)),
    }
    if (config.get("model_type"), tuple(config.get(key) for key in keys)) != expected[role]:
        raise ValueError(f"Wrong {role} architecture: expected MiMo-7B teacher and Qwen3-1.7B student")
    if not 2 <= max_context <= config.get("max_position_embeddings", 0):
        raise ValueError(f"Requested context exceeds {role} capacity")
    vocab = token_map(path)
    if any(type(index) is not int or not 0 <= index < config["vocab_size"] for index in vocab.values()):
        raise ValueError(f"Invalid {role} tokenizer IDs")
    index_path = path / "model.safetensors.index.json"
    shards = (
        sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        if index_path.is_file() else ["model.safetensors"]
    )
    if not shards or any(Path(name).name != name for name in shards):
        raise ValueError(f"Invalid {role} weight index")
    weights = {}
    for name in shards:
        shard = path / name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"Missing or empty {role} weight shard: {shard}")
        stat = shard.stat()
        weights[name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if role == "teacher":
        files += ["configuration_mimo.py", "modeling_mimo.py"]
    files += [name for name in ("generation_config.json", "model.safetensors.index.json") if (path / name).is_file()]
    metadata = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in files}
    generation = path / "generation_config.json"
    generation = json.loads(generation.read_text()) if generation.is_file() else {}
    stops = generation.get("eos_token_id", config.get("eos_token_id"))
    stops = [stops] if type(stops) is int else stops
    if not isinstance(stops, list) or not stops or any(
        type(token) is not int or not 0 <= token < MIMO_SHARED_VOCAB_SIZE for token in stops
    ):
        raise ValueError(f"{role} EOS tokens must belong to the shared vocabulary")
    return dict(path=str(path), metadata_sha256=metadata, weights=weights)


def record_protocol(save_dir, student, teacher):
    directory = Path(save_dir)
    manifest = directory / "mimo_opd.json"
    protocol = dict(
        method="vanilla_opd", shared_vocab_size=MIMO_SHARED_VOCAB_SIZE,
        prompt_version=SHARED_PROMPT_VERSION, student=student, teacher=teacher,
    )
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("MiMo OPD model/tokenizer or protocol changed; use a new SAVE_DIR")
        return
    if (directory / "latest_checkpointed_iteration.txt").exists() or any(directory.glob("iter_*")):
        raise ValueError("Cannot resume a checkpoint without its MiMo shared-vocabulary manifest")
    directory.mkdir(parents=True, exist_ok=True)
    with manifest.open("x") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True)
        stream.write("\n")


class MiMoTeacherScorer:
    def __init__(self, model, prompt_adapter, max_context, chunk_size):
        self.model = model.eval().requires_grad_(False)
        self.prompt_adapter = prompt_adapter
        self.max_context = max_context
        self.chunk_size = chunk_size

    @torch.inference_mode()
    def score(self, request):
        if not isinstance(request, dict) or set(request) != {"input_ids", "response_length"}:
            raise ValueError("Expected input_ids and response_length")
        tokens, response_length = request["input_ids"], request["response_length"]
        if (
            not isinstance(tokens, list) or any(type(token) is not int for token in tokens)
            or type(response_length) is not int or not 0 <= response_length < len(tokens) <= self.max_context
        ):
            raise ValueError("Invalid token sequence or prompt/response length")
        prompt_length = len(tokens) - response_length
        response = tokens[prompt_length:]
        if any(token < 0 or token >= MIMO_SHARED_VOCAB_SIZE for token in response):
            raise ValueError("Response token outside the shared vocabulary")
        prompt = self.prompt_adapter(tokens[:prompt_length])
        if len(prompt) + response_length > self.model.config.max_position_embeddings:
            raise ValueError("Retokenized prompt/response exceeds teacher context capacity")
        if not response_length:
            return {"teacher_log_probs": []}

        device = next(self.model.parameters()).device
        teacher_tokens = torch.tensor(prompt + response, dtype=torch.long, device=device)
        hidden = self.model.model(
            input_ids=teacher_tokens[:-1].unsqueeze(0), use_cache=False, return_dict=True
        ).last_hidden_state[0, len(prompt) - 1:]
        targets = teacher_tokens[len(prompt):]
        scores = []
        for start in range(0, response_length, self.chunk_size):
            stop = min(start + self.chunk_size, response_length)
            logits = self.model.lm_head(hidden[start:stop]).float()[:, :MIMO_SHARED_VOCAB_SIZE]
            log_probs = logits.gather(-1, targets[start:stop, None]).squeeze(-1) - logits.logsumexp(-1)
            scores.append(log_probs.cpu())
        values = torch.cat(scores)
        if not torch.isfinite(values).all():
            raise RuntimeError("Non-finite MiMo teacher log-probabilities")
        return {"teacher_log_probs": values.tolist()}


def create_app(scorer):
    app = web.Application(client_max_size=max(1024**2, scorer.max_context * 32))
    lock = asyncio.Lock()

    async def health(request):
        return web.json_response(dict(status="ready", shared_vocab_size=MIMO_SHARED_VOCAB_SIZE))

    async def score(request):
        # Serialize request parsing and forwards to bound peak memory.
        async with lock:
            try:
                payload = await request.json()
                result = await asyncio.to_thread(scorer.score, payload)
            except (ValueError, KeyError, TypeError) as error:
                raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(result)

    app.router.add_get("/health_generate", health)
    app.router.add_post("/score", score)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13141)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    student = model_metadata(args.student_model, "student", args.max_context)
    teacher = model_metadata(args.teacher_model, "teacher", args.max_context)
    validate_tokenizers([args.student_model, args.teacher_model])
    adapter = MiMoPromptAdapter(
        AutoTokenizer.from_pretrained(args.student_model, local_files_only=True),
        AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True, trust_remote_code=True),
    )
    record_protocol(args.save_dir, student, teacher)
    logger.info("Loading frozen MiMo teacher from %s", args.teacher_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, local_files_only=True,
    ).to("cuda:0")
    scorer = MiMoTeacherScorer(model, adapter, args.max_context, args.chunk_size)
    web.run_app(create_app(scorer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
