"""Sparse stand-ins for TensorFold/Qwen3.8-27B-MLX-4bit and z-lab/Qwen3.8-27B-DFlash2.

Every tensor has its real name, dtype and shape, and its bytes are a hole in a sparse file (zeros), so the
checkpoints cost no disk and no download. The geometry is the 27B's (TensorFold's own fixtures:
zig/src/families/qwen27/testdata/gguf-qwen35-config.json and the HIP drafter's config test).
"""
import json, os, sys

def write_safetensors(path, tensors):
    """tensors: list of (name, dtype, shape). Writes the header; the data is a sparse hole."""
    size = {"BF16": 2, "U32": 4, "F32": 4}
    header, off = {}, 0
    for name, dt, shape in tensors:
        n = size[dt]
        for d in shape:
            n *= d
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "mlx"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        f.truncate(8 + len(h) + off)
    return off

H, I, LAYERS, V = 5120, 17408, 64, 248320
HEADS, KV, HD = 24, 4, 256
LK, LV, LKD, LVD, CONV = 16, 48, 128, 128, 4
BITS, GROUP = 4, 64

def affine(out, name, n, k):
    out.append((name + ".weight", "U32", (n, k * BITS // 32)))
    out.append((name + ".scales", "BF16", (n, k // GROUP)))
    out.append((name + ".biases", "BF16", (n, k // GROUP)))

def target(d):
    os.makedirs(d, exist_ok=True)
    L = "language_model.model."
    t = []
    affine(t, L + "embed_tokens", V, H)
    t.append((L + "norm.weight", "BF16", (H,)))
    affine(t, "language_model.lm_head", V, H)
    types = []
    for i in range(LAYERS):
        b = f"{L}layers.{i}."
        t.append((b + "input_layernorm.weight", "BF16", (H,)))
        t.append((b + "post_attention_layernorm.weight", "BF16", (H,)))
        affine(t, b + "mlp.gate_proj", I, H)
        affine(t, b + "mlp.up_proj", I, H)
        affine(t, b + "mlp.down_proj", H, I)
        if (i + 1) % 4 == 0:
            types.append("full_attention")
            a = b + "self_attn."
            affine(t, a + "q_proj", HEADS * HD * 2, H)
            affine(t, a + "k_proj", KV * HD, H)
            affine(t, a + "v_proj", KV * HD, H)
            affine(t, a + "o_proj", H, HEADS * HD)
            t.append((a + "q_norm.weight", "BF16", (HD,)))
            t.append((a + "k_norm.weight", "BF16", (HD,)))
        else:
            types.append("linear_attention")
            a = b + "linear_attn."
            qkv = 2 * LK * LKD + LV * LVD
            affine(t, a + "in_proj_qkv", qkv, H)
            affine(t, a + "in_proj_z", LV * LVD, H)
            affine(t, a + "in_proj_a", LV, H)
            affine(t, a + "in_proj_b", LV, H)
            t.append((a + "conv1d.weight", "BF16", (qkv, CONV, 1)))
            t.append((a + "A_log", "BF16", (LV,)))
            t.append((a + "dt_bias", "BF16", (LV,)))
            t.append((a + "norm.weight", "BF16", (LVD,)))
            affine(t, a + "out_proj", H, LV * LVD)
    n = write_safetensors(os.path.join(d, "model.safetensors"), t)
    cfg = {
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"], "tie_word_embeddings": False,
        "quantization": {"group_size": GROUP, "bits": BITS, "mode": "affine"},
        "text_config": {
            "hidden_size": H, "intermediate_size": I, "num_hidden_layers": LAYERS, "vocab_size": V,
            "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD,
            "linear_num_key_heads": LK, "linear_num_value_heads": LV, "linear_key_head_dim": LKD,
            "linear_value_head_dim": LVD, "linear_conv_kernel_dim": CONV, "rms_norm_eps": 1e-6,
            "full_attention_interval": 4, "mtp_num_hidden_layers": 0, "eos_token_id": 248046,
            "max_position_embeddings": 262144, "layer_types": types,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        },
    }
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    return n

def drafter(d):
    os.makedirs(d, exist_ok=True)
    taps, layers, heads, kv, hd, rank, group, window = [5, 19, 33, 47, 61], 5, 32, 8, 128, 256, 16, 2048
    dyn = 4 * (H // group)
    t = [("fc.weight", "BF16", (H, len(taps) * H)), ("hidden_norm.weight", "BF16", (H,)), ("norm.weight", "BF16", (H,)),
         ("candidate_selector.hidden_projection.weight", "BF16", (rank, H)),
         ("candidate_selector.predecessor_codebook", "BF16", (V, rank)),
         ("candidate_selector.successor_codebook", "BF16", (V, rank))]
    for i in range(layers):
        b = f"layers.{i}."
        t += [(b + "input_layernorm.weight", "BF16", (H,)), (b + "post_attention_layernorm.weight", "BF16", (H,)),
              (b + "self_attn.q_proj.weight", "BF16", (heads * hd, H)), (b + "self_attn.k_proj.weight", "BF16", (kv * hd, H)),
              (b + "self_attn.v_proj.weight", "BF16", (kv * hd, H)), (b + "self_attn.o_proj.weight", "BF16", (H, heads * hd)),
              (b + "self_attn.q_norm.weight", "BF16", (hd,)), (b + "self_attn.k_norm.weight", "BF16", (hd,)),
              (b + "mlp.gate_proj.weight", "BF16", (I, H)), (b + "mlp.up_proj.weight", "BF16", (I, H)),
              (b + "mlp.down_proj.weight", "BF16", (H, I)),
              (b + "attention_conv.kernel_projection.weight", "BF16", (dyn, H)), (b + "attention_conv.base_kernel", "BF16", (2, 2, H)),
              (b + "mlp_conv.kernel_projection.weight", "BF16", (dyn, H)), (b + "mlp_conv.base_kernel", "BF16", (2, 2, H))]
    n = write_safetensors(os.path.join(d, "model.safetensors"), t)
    cfg = {"dflash_config": {"block_size": 8, "conv_group_size": group, "conv_kernel_size": 2, "mask_token_id": 248070,
                             "selector_rank": rank, "selector_top_k": 16, "target_layer_ids": taps},
           "head_dim": hd, "hidden_size": H, "intermediate_size": I, "num_attention_heads": heads,
           "num_hidden_layers": layers, "num_key_value_heads": kv, "rms_norm_eps": 1e-6,
           "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "sliding_window": window, "vocab_size": V}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    return n

def tokenizer(d):
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    vocab = {chr(c): b for b, c in zip(bs, cs)}
    # placeholders up to the model's vocabulary, so the EOS ids are inside the tokenizer's range
    for i in range(256, V):
        if i not in (248044, 248045, 248046):
            vocab[f"<|pad{i}|>"] = i
    specials = {"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046}
    added = [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
             for s, i in specials.items()]
    tok = {"version": "1.0", "added_tokens": added, "normalizer": None,
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": True},
           "post_processor": None, "decoder": {"type": "ByteLevel"},
           "model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": False, "ignore_merges": False}}
    json.dump(tok, open(os.path.join(d, "tokenizer.json"), "w"))
    json.dump({"eos_token": "<|im_end|>", "bos_token": None,
               "chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"},
              open(os.path.join(d, "tokenizer_config.json"), "w"))

if __name__ == "__main__":
    root = sys.argv[1]
    nt = target(os.path.join(root, "Qwen3.8-27B-MLX-4bit"))
    tokenizer(os.path.join(root, "Qwen3.8-27B-MLX-4bit"))
    nd = drafter(os.path.join(root, "Qwen3.8-27B-DFlash2"))
    print(f"target {nt / 2**30:.2f} GiB, drafter {nd / 2**30:.2f} GiB (sparse)")
