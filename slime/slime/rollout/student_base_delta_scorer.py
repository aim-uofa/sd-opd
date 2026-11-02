"""Score the base, optional teacher and initial student on current-actor IDs."""

import argparse
import asyncio
import hashlib
import io
import json
import logging

import numpy as np
import torch
from aiohttp import web

from slime.utils.distillation_scoring import score_model
from slime.utils.student_base_delta import delta_signals, identity
from slime.utils.student_base_delta_config import validate_models
from slime.utils.shared_vocab import MIMO_SHARED_VOCAB_SIZE, QWEN3_THINK_TOKEN_IDS, MiMoPromptAdapter

logger = logging.getLogger(__name__)


def decode_request(body, vocab_size, top_k, max_context, *, prompt_token_ids=()):
    with np.load(io.BytesIO(body), allow_pickle=False) as arrays:
        if set(arrays.files) != {"tokens", "student", "support_ids", "support_logps", "vocab_size"}:
            raise ValueError("Unexpected student-base delta request fields.")
        vocab = arrays["vocab_size"]
        if vocab.shape != () or vocab.dtype.kind not in "iu" or int(vocab) != vocab_size:
            raise ValueError("Actor and scorer vocabulary sizes differ (including padding).")
        if any(arrays[key].dtype.kind not in "iu" for key in ("tokens", "support_ids")):
            raise ValueError("Token IDs must be integers.")
        if any(arrays[key].dtype.kind != "f" for key in ("student", "support_logps")):
            raise ValueError("Actor log-probabilities must be floating point.")
        tokens = torch.from_numpy(arrays["tokens"].copy()).long()
        ids = torch.from_numpy(arrays["support_ids"].copy()).long()
        student = torch.from_numpy(arrays["student"].copy()).float()
        logps = torch.from_numpy(arrays["support_logps"].copy()).float()
    if tokens.ndim != 1 or student.ndim != 1 or not 0 < len(student) < len(tokens) <= max_context:
        raise ValueError("Invalid prompt/response lengths or context limit exceeded.")
    if ids.shape != (len(student), top_k) or ids.shape != logps.shape:
        raise ValueError("Student support shape or top-k differs from scorer configuration.")
    prompt, response = tokens[:-len(student)], tokens[-len(student):]
    invalid_prompt = (prompt < 0) | (prompt >= vocab_size)
    for token in prompt_token_ids:
        invalid_prompt &= prompt != token
    if invalid_prompt.any() or response.min() < 0 or response.max() >= vocab_size or ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("Token ID outside the vocabulary.")
    if any(not torch.isfinite(x).all() or (x > 1e-5).any() for x in (student, logps)):
        raise ValueError("Invalid student log-probabilities.")
    chosen = ids == tokens[-len(student):, None]
    if not (chosen.sum(-1) == 1).all() or not torch.allclose(logps[chosen], student, atol=2e-4, rtol=2e-4):
        raise ValueError("Sampled tokens and student support probabilities are misaligned.")
    ordered = ids.sort(-1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any() or (logps.exp().sum(-1) > 1.0001).any():
        raise ValueError("Duplicate candidate IDs or invalid probability mass.")
    return tokens, student, ids, logps


class StudentBaseDeltaScorer:
    def __init__(self, teacher, student_base, *, mode="boost", strength=0.25,
                 top_k=1024, chunk_size=128, max_context=8192, initial_student=None, shared_vocab_size=0,
                 teacher_prompt_adapter=None):
        self.identity = identity(mode, strength, top_k, "initial" if initial_student is not None else "current",
                                 shared_vocab_size)
        if (mode == "ungated") != (teacher is None):
            raise ValueError("Provide a teacher for direct/boost/vanilla, and none for ungated scoring.")
        if (mode == "vanilla") != (student_base is None) or (mode == "vanilla" and initial_student is not None):
            raise ValueError("Vanilla requires no base/initial student; other modes require a base")
        self.teacher = teacher.eval().requires_grad_(False) if teacher is not None else None
        self.student_base = student_base.eval().requires_grad_(False) if student_base is not None else None
        self.initial_student = initial_student.eval().requires_grad_(False) if initial_student is not None else None
        self.shared_vocab_size = shared_vocab_size
        if shared_vocab_size == MIMO_SHARED_VOCAB_SIZE and teacher_prompt_adapter is None:
            raise ValueError('MiMo scoring requires model-specific prompt tokenization')
        self.teacher_prompt_adapter = teacher_prompt_adapter
        reference = student_base if student_base is not None else teacher
        self.vocab_size = shared_vocab_size or reference.config.vocab_size
        models = [m for m in (teacher, student_base, initial_student) if m is not None]
        if (self.vocab_size < 1 or any(m.config.vocab_size < self.vocab_size or
                (not shared_vocab_size and m.config.vocab_size != self.vocab_size) for m in models)
            or not 1 <= top_k <= self.vocab_size
            or chunk_size < 1 or max_context < 2):
            raise ValueError("Invalid scorer vocabulary, top-k, chunk size or context limit.")
        self.chunk_size, self.max_context = chunk_size, max_context

    @torch.inference_mode()
    def score(self, body):
        tokens, student, ids, logps = decode_request(
            body, self.vocab_size, self.identity["top_k"], self.max_context,
            prompt_token_ids=QWEN3_THINK_TOKEN_IDS if self.shared_vocab_size == MIMO_SHARED_VOCAB_SIZE else (),
        )
        teacher, teacher_support = None, None
        if self.teacher is not None:
            teacher_tokens = tokens
            if self.teacher_prompt_adapter is not None:
                prompt_ids = self.teacher_prompt_adapter(tokens[:-len(student)].tolist())
                teacher_tokens = torch.cat((torch.tensor(prompt_ids, dtype=tokens.dtype), tokens[-len(student):]))
                # The teacher prompt can be longer because its thinking tags use BPE tokens.
                if len(teacher_tokens) > self.teacher.config.max_position_embeddings:
                    raise ValueError('Retokenized prompt/response exceeds teacher context capacity')
            teacher, teacher_support = score_model(self.teacher, teacher_tokens, len(student), ids, self.chunk_size,
                                                   self.shared_vocab_size)
        base, base_support = None, None
        if self.student_base is not None:
            base, base_support = score_model(self.student_base, tokens, len(student), ids, self.chunk_size,
                                             self.shared_vocab_size)
        initial, initial_support = None, None
        if self.initial_student is not None:
            initial, initial_support = score_model(self.initial_student, tokens, len(student), ids, self.chunk_size,
                                                   self.shared_vocab_size)
        result = delta_signals(student, teacher, base, logps, teacher_support, base_support,
                               mode=self.identity["mode"], strength=self.identity["strength"],
                               initial_student=initial, initial_student_support=initial_support)
        metadata = dict(self.identity, vocab_size=self.vocab_size, request_sha256=hashlib.sha256(body).hexdigest())
        buffer = io.BytesIO()
        np.savez(buffer, **{key: value.cpu().numpy() for key, value in result.items()},
                 metadata=np.frombuffer(json.dumps(metadata, allow_nan=False, sort_keys=True).encode(), dtype=np.uint8))
        return buffer.getvalue()


def create_app(scorer):
    app = web.Application(client_max_size=512 * 1024**2)
    lock = asyncio.Lock()

    async def health(request):
        return web.json_response(dict(status="ready", vocab_size=scorer.vocab_size, **scorer.identity))

    async def score(request):
        async with lock:
            body = await request.read()
            try:
                result = await asyncio.to_thread(scorer.score, body)
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise web.HTTPBadRequest(text=str(error)) from error
            except Exception:
                logger.exception("Student-base delta scoring failed; no fallback signal was returned")
                raise
        return web.Response(body=result, content_type="application/octet-stream")

    app.router.add_get("/health_generate", health)
    app.router.add_post("/score", score)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-base-model", help="Required except for vanilla.")
    parser.add_argument("--teacher-model", help="Required for direct/boost/vanilla; unused for ungated.")
    parser.add_argument("--student-source", choices=("current", "initial"), default="current",
                        help="initial loads a frozen STUDENT_MODEL copy on the base device for the base log-ratio.")
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--base-device", default="cuda:0")
    parser.add_argument("--mode", choices=("direct", "boost", "ungated", "vanilla"), required=True)
    parser.add_argument("--shared-vocab-size", type=int, default=0)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13912)
    args = parser.parse_args()
    identity(args.mode, args.strength, args.top_k, args.student_source, args.shared_vocab_size)
    if args.mode != "vanilla" and not args.student_base_model:
        parser.error("--student-base-model is required except for vanilla")
    if args.shared_vocab_size and (args.mode not in ("vanilla", "direct") or args.student_source != "current"):
        parser.error("MiMo shared vocabulary requires current-student vanilla/direct")
    if args.mode == "vanilla" and (not args.shared_vocab_size or args.top_k != 1):
        parser.error("vanilla requires shared vocabulary and top-k=1")
    if args.mode != "ungated" and not args.teacher_model:
        parser.error("--teacher-model is required for direct/boost/vanilla")
    teacher_path = args.teacher_model if args.mode != "ungated" else None
    base_path = args.student_base_model if args.mode != "vanilla" else None
    config = validate_models(args.student_model, teacher_path, base_path,
                             max_context=args.max_context, shared_vocab_size=args.shared_vocab_size)
    if args.top_k > config["vocab_size"] or args.chunk_size < 1:
        parser.error("invalid top-k or chunk size")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher_prompt_adapter = None
    if args.shared_vocab_size:
        teacher_prompt_adapter = MiMoPromptAdapter(
            AutoTokenizer.from_pretrained(args.student_model, local_files_only=True),
            AutoTokenizer.from_pretrained(teacher_path, local_files_only=True, trust_remote_code=True),
        )

    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    models = {}
    paths = {"student_base": (base_path, args.base_device)} if base_path is not None else {}
    if teacher_path is not None:
        paths = {"teacher": (teacher_path, args.teacher_device), **paths}
    if args.student_source == "initial":
        paths["initial_student"] = (args.student_model, args.base_device)
    for role, (path, device) in paths.items():
        logger.info("Loading frozen %s on %s", path, device)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        models[role] = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="sdpa", local_files_only=True,
            trust_remote_code=bool(args.shared_vocab_size and role == "teacher"),
        ).to(device)
    scorer = StudentBaseDeltaScorer(models.get("teacher"), models.get("student_base"),
                                    mode=args.mode, strength=args.strength, top_k=args.top_k,
                                    chunk_size=args.chunk_size, max_context=args.max_context,
                                    initial_student=models.get("initial_student"),
                                    shared_vocab_size=args.shared_vocab_size,
                                    teacher_prompt_adapter=teacher_prompt_adapter)
    web.run_app(create_app(scorer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
