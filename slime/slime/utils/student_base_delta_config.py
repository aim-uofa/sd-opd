"""Local model checks without importing torch, loading weights, or using GPUs."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_map(path):
    data = json.loads((path / "tokenizer.json").read_text())
    vocab = dict(data["model"]["vocab"])
    for token in data.get("added_tokens", []):
        vocab[token["content"]] = token["id"]
    config = json.loads((path / "tokenizer_config.json").read_text())
    for index, token in config.get("added_tokens_decoder", {}).items():
        if token["content"] in vocab and vocab[token["content"]] != int(index):
            raise ValueError(f"Conflicting added token ID in {path}")
        vocab[token["content"]] = int(index)
    return vocab


def validate_models(student, teacher, student_base, *, max_context=8192, shared_vocab_size=0):
    models = {"student": student}
    if teacher is not None:
        models["teacher"] = teacher
    if student_base is not None:
        models["student_base"] = student_base
    elif not shared_vocab_size:
        raise ValueError("A student base is required outside shared-vocabulary vanilla OPD")
    if shared_vocab_size:
        from slime.utils.shared_vocab import MIMO_SHARED_VOCAB_SIZE, validate_tokenizers

        if shared_vocab_size != MIMO_SHARED_VOCAB_SIZE or teacher is None:
            raise ValueError("Shared-vocabulary mode requires the MiMo teacher and 151665 shared tokens")
    roles = tuple(models)
    paths = [Path(p).resolve() for p in models.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("Student, student base and optional teacher must be separate local model directories.")
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "vocab_size")
    dense_shapes = ((2048, 6144, 28, 16, 8, 151936), (2560, 9728, 36, 32, 8, 151936),
                    (4096, 12288, 36, 32, 8, 151936))
    moe_shape = (2048, 6144, 48, 32, 4, 151936)
    fingerprints, configs, vocabs, missing_weights = {}, {}, [], []
    for role, path in zip(roles, paths, strict=True):
        config = json.loads((path / "config.json").read_text())
        shape = tuple(config.get(key) for key in keys)
        if role == "teacher":
            valid = (config.get("model_type") == "qwen3" and shape in dense_shapes[1:]) or (
                config.get("model_type") == "qwen3_moe" and shape == moe_shape
                and config.get("moe_intermediate_size") == 768
                and config.get("num_experts") == 128 and config.get("num_experts_per_tok") == 8
            )
            if shared_vocab_size:
                valid = config.get("model_type") == "mimo" and shape == (4096, 11008, 36, 32, 8, 151680)
                if not all((path / name).is_file() for name in ("configuration_mimo.py", "modeling_mimo.py")):
                    raise ValueError(f"Missing local MiMo model code: {path}")
        else:
            valid = config.get("model_type") == "qwen3" and shape in dense_shapes
            if shared_vocab_size:
                valid = valid and shape == dense_shapes[0]
            if role == "student_base":
                valid = valid and shape == tuple(configs["student"].get(key) for key in keys)
        if not valid:
            if shared_vocab_size:
                raise ValueError(f"Wrong {role} architecture: MiMo-7B teacher, Qwen3-1.7B student/base required")
            raise ValueError(
                f"Wrong {role} architecture: expected a Qwen3-1.7B/4B/8B student with a matching student base, "
                "and a Qwen3-4B-Instruct-2507, Qwen3-8B or Qwen3-30B-A3B-Instruct-2507 teacher."
            )
        if not 2 <= max_context <= config.get("max_position_embeddings", 0):
            raise ValueError(f"Context exceeds {role} model capacity.")
        index = path / "model.safetensors.index.json"
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
        if not shards or any(Path(name).name != name for name in shards):
            raise ValueError(f"Invalid {role} weight index: {path}")
        missing = [name for name in shards if not (path / name).is_file() or (path / name).stat().st_size == 0]
        if missing:
            raise ValueError(f"Missing or invalid {role} weight shards: {path}")
        missing_weights.extend(str(path / name) for name in missing)
        vocab = token_map(path)
        if not vocab or any(type(i) is not int or not 0 <= i < shape[-1] for i in vocab.values()):
            raise ValueError(f"Invalid {role} tokenizer ID range.")
        vocabs.append(vocab)
        configs[role] = config
        metadata = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                    "model.safetensors.index.json")
        if shared_vocab_size and role == "teacher":
            metadata += ("configuration_mimo.py", "modeling_mimo.py")
        fingerprints[role] = dict(
            path=str(path), metadata_sha256={name: sha256(path / name) for name in metadata if (path / name).is_file()},
            weights={name: dict(bytes=(path / name).stat().st_size, mtime_ns=(path / name).stat().st_mtime_ns)
                     for name in shards if name not in missing},
        )
    if shared_vocab_size:
        validate_tokenizers(paths)
        from slime.utils.shared_vocab import QWEN3_THINK_TOKEN_IDS

        for role, vocab in zip(roles, vocabs, strict=True):
            if role != "teacher" and tuple(vocab.get(t) for t in ("<think>", "</think>")) != QWEN3_THINK_TOKEN_IDS:
                raise ValueError(f"Unexpected {role} thinking token IDs")
    elif any(vocab != vocabs[0] for vocab in vocabs[1:]):
        raise ValueError("Student, teacher and student base must have identical token-to-ID mappings.")
    generation = paths[0] / "generation_config.json"
    generation = json.loads(generation.read_text()) if generation.is_file() else {}
    stops = generation.get("eos_token_id", configs["student"].get("eos_token_id"))
    stops = [stops] if type(stops) is int else stops
    if (not isinstance(stops, list) or not stops
        or any(type(i) is not int or not 0 <= i < configs["student"]["vocab_size"] for i in stops)
        or len(set(stops)) != len(stops)):
        raise ValueError("Student must declare valid unique EOS token IDs.")
    if shared_vocab_size and any(i >= shared_vocab_size for i in stops):
        raise ValueError("Student EOS tokens must belong to the shared vocabulary")
    result = dict(vocab_size=shared_vocab_size or configs["student"]["vocab_size"],
                  stop_token_ids=sorted(stops), model_fingerprints=fingerprints)
    if shared_vocab_size:
        result["missing_weights"] = missing_weights
    return result
