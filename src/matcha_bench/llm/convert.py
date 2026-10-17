"""Convert a ChatGLM-format GLM-4 checkpoint (e.g. zwzzz/MentalGLM-chat) to the native
transformers `glm` format, so it loads without `trust_remote_code` on any transformers >= 4.46
and works with peft and bitsandbytes.

The released MentalGLM models ship the original THUDM code (written for transformers 4.43), which
no longer runs on current transformers. The native `GlmForCausalLM` is the same network with
renamed and split tensors:

* `query_key_value` is split into q/k/v (32 query heads, 2 key/value heads, head_dim 128);
* the MLP, norms, embedding and output layer only change names;
* RoPE: both rotate the first half of each head (partial_rotary_factor 0.5) on interleaved
  pairs, with base 10000 * rope_ratio. rope_ratio is 500 for the chat models (MentalGLM-chat,
  glm-4-9b-chat) and 1 for the base models, so rope_theta must be set from it.

The tokenizer is identical to the native one (token ids checked on Chinese text and every
special token), so the native `tokenizer.json` is used with the source model's chat template.
Shards are converted one at a time and the source shard is deleted afterwards when asked, so a
9B model needs about 20 GB of free disk, not 40.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..utils import get_logger

log = get_logger()

# Native tokenizer with the same vocabulary (THUDM's own conversion of glm-4-9b-chat).
TOKENIZER_REPO = "zai-org/glm-4-9b-chat-hf"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]  # all that repo has; both required

_LAYER = re.compile(r"^transformer\.encoder\.layers\.(\d+)\.(.+)$")
_RENAME = {
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "self_attention.dense.weight": "self_attn.o_proj.weight",
    "mlp.dense_h_to_4h.weight": "mlp.gate_up_proj.weight",
    "mlp.dense_4h_to_h.weight": "mlp.down_proj.weight",
}
_TOP = {
    "transformer.embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "transformer.encoder.final_layernorm.weight": "model.norm.weight",
    "transformer.output_layer.weight": "lm_head.weight",
}


def glm_config_from_chatglm(src: dict, dtype: str = "float16") -> dict:
    """Native `glm` config.json for a ChatGLM config.json (GLM-4 generation)."""
    n_heads = src["num_attention_heads"]
    head_dim = src.get("kv_channels") or src["hidden_size"] // n_heads
    n_kv = src["multi_query_group_num"] if src.get("multi_query_attention") else n_heads
    if not src.get("original_rope", True):
        raise ValueError("only original_rope=True (GLM-4) checkpoints are supported")
    eos = src.get("eos_token_id")
    return {
        "architectures": ["GlmForCausalLM"],
        "model_type": "glm",
        "vocab_size": src["padded_vocab_size"],
        "hidden_size": src["hidden_size"],
        "intermediate_size": src["ffn_hidden_size"],
        "num_hidden_layers": src["num_layers"],
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "attention_bias": bool(src.get("add_qkv_bias", True)),
        "attention_dropout": 0.0,
        "max_position_embeddings": src.get("seq_length", 8192),
        "rms_norm_eps": src["layernorm_epsilon"],
        # legacy keys: transformers 4.x reads them directly, 5.x converts them to rope_parameters
        "rope_theta": 10000.0 * float(src.get("rope_ratio", 1) or 1),
        "partial_rotary_factor": 0.5,
        "tie_word_embeddings": False,
        "pad_token_id": src.get("pad_token_id", 151329),
        "eos_token_id": eos if eos is not None else [151329, 151336, 151338],
        "use_cache": True,
        "torch_dtype": dtype,
    }


def convert_tensors(tensors: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    """Rename (and split) the tensors of one ChatGLM shard into native `glm` names."""
    q_size = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_size = cfg["num_key_value_heads"] * cfg["head_dim"]
    out = {}
    for name, t in tensors.items():
        if name.endswith("rotary_pos_emb.inv_freq"):
            continue  # recomputed from rope_theta
        if name in _TOP:
            out[_TOP[name]] = t
            continue
        m = _LAYER.match(name)
        if not m:
            raise KeyError(f"unexpected tensor {name}")
        i, rest = m.groups()
        prefix = f"model.layers.{i}."
        if rest.startswith("self_attention.query_key_value."):
            kind = rest.rsplit(".", 1)[1]  # weight or bias
            if t.shape[0] != q_size + 2 * kv_size:
                raise ValueError(f"{name}: expected {q_size + 2 * kv_size} rows, got {t.shape[0]}")
            q, k, v = t.split([q_size, kv_size, kv_size], dim=0)
            out[prefix + f"self_attn.q_proj.{kind}"] = q.contiguous()
            out[prefix + f"self_attn.k_proj.{kind}"] = k.contiguous()
            out[prefix + f"self_attn.v_proj.{kind}"] = v.contiguous()
        elif rest in _RENAME:
            out[prefix + _RENAME[rest]] = t
        else:
            raise KeyError(f"unexpected tensor {name}")
    return out


def _source_files(source: str, cache_dir: Path | None):
    """(config dict, tokenizer_config dict, list of shard names, fetch(name) -> local path)."""
    local = Path(source)
    if local.is_dir():
        def fetch(name):
            return local / name
    else:
        from huggingface_hub import hf_hub_download

        def fetch(name):
            return Path(hf_hub_download(source, name, cache_dir=cache_dir))
    config = json.loads(fetch("config.json").read_text(encoding="utf-8"))
    tok_cfg = json.loads(fetch("tokenizer_config.json").read_text(encoding="utf-8"))
    try:
        index = json.loads(fetch("model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    except Exception:
        shards = ["model.safetensors"]
    return config, tok_cfg, shards, fetch


def _copy_tokenizer(out: Path, tokenizer_source: str, chat_template: str | None, cache_dir: Path | None) -> None:
    local = Path(tokenizer_source)
    for name in TOKENIZER_FILES:
        if local.is_dir():
            src = local / name
        else:
            from huggingface_hub import hf_hub_download
            src = Path(hf_hub_download(tokenizer_source, name, cache_dir=cache_dir))
        shutil.copy2(src, out / name)
    if chat_template:
        # keep the source model's own template (the one it was fine-tuned with)
        cfg_path = out / "tokenizer_config.json"
        tok_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tok_cfg["chat_template"] = chat_template
        cfg_path.write_text(json.dumps(tok_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "chat_template.jinja").unlink(missing_ok=True)


def ensure_converted(source: str, convert_dir: str | Path, hf_cache: str | Path | None = None) -> str:
    """Path of a native-format copy of `source`, converting it first if needed.

    A local folder that is already in native format (model_type glm) is used as it is.
    """
    local = Path(source)
    if local.is_dir() and json.loads((local / "config.json").read_text(encoding="utf-8")).get("model_type") == "glm":
        return str(local)
    out = Path(convert_dir) / source.strip("/").replace("/", "__").replace(":", "")
    return str(convert_chatglm_checkpoint(source, out, delete_source_shards=True, cache_dir=hf_cache))


def convert_chatglm_checkpoint(
    source: str,
    out_dir: str | Path,
    dtype: str = "float16",
    tokenizer_source: str = TOKENIZER_REPO,
    delete_source_shards: bool = False,
    cache_dir: str | Path | None = None,
) -> Path:
    """Convert `source` (a hub id or a local ChatGLM checkpoint folder) into `out_dir`.

    Skips the work if `out_dir` already holds a finished conversion. `dtype` is the dtype the
    weights are saved in (float16 for T4 GPUs, which have no fast bfloat16).
    """
    out = Path(out_dir)
    if (out / ".converted").exists():
        log.info("Already converted: %s", out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else None
    src_cfg, src_tok_cfg, shards, fetch = _source_files(source, cache)
    if src_cfg.get("model_type") != "chatglm":
        raise ValueError(f"{source} is not a ChatGLM checkpoint (model_type={src_cfg.get('model_type')})")
    cfg = glm_config_from_chatglm(src_cfg, dtype)
    torch_dtype = getattr(torch, dtype)

    # tokenizer first: a problem there fails in seconds, not after a 19 GB download
    _copy_tokenizer(out, tokenizer_source, src_tok_cfg.get("chat_template"), cache)

    weight_map, total = {}, 0
    for n, shard in enumerate(shards, 1):
        path = fetch(shard)
        with safe_open(str(path), framework="pt") as f:
            tensors = {k: f.get_tensor(k).to(torch_dtype) for k in f.keys()}
        converted = convert_tensors(tensors, cfg)
        del tensors
        out_name = f"model-{n:05d}-of-{len(shards):05d}.safetensors"
        save_file(converted, str(out / out_name), metadata={"format": "pt"})
        for k, t in converted.items():
            weight_map[k] = out_name
            total += t.numel() * t.element_size()
        del converted
        if delete_source_shards and not Path(source).is_dir():
            Path(path).resolve().unlink(missing_ok=True)  # hub cache keeps a symlink to the blob
        log.info("converted %s (%d/%d)", shard, n, len(shards))

    # embed/norm/lm_head + per layer: 2 norms, q/k/v (+ biases), o_proj, gate_up, down
    expected = 3 + (11 if cfg["attention_bias"] else 8) * cfg["num_hidden_layers"]
    if len(weight_map) != expected:
        raise ValueError(f"converted {len(weight_map)} tensors, expected {expected}")
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2),
        encoding="utf-8")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    gen = {"do_sample": False, "eos_token_id": cfg["eos_token_id"], "pad_token_id": cfg["pad_token_id"]}
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")
    (out / ".converted").write_text(json.dumps({"source": source, "dtype": dtype}), encoding="utf-8")
    log.info("Converted %s -> %s (%.1f GB)", source, out, total / 1e9)
    return out
