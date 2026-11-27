#!/usr/bin/env python3
"""Numpy reference of the DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2, sglang srt/models/dflash.py DFlash2DraftModel) against the engine's dump (INK2_DF2_DUMP=<dir>).
usage: ref_check.py <model.safetensors> <dump dir>      stages compared: context K / V, embeddings, per-layer block output, final hidden, candidates, drafts."""
import json, struct, sys
import numpy as np
H, HD, NQ, NKV, FF, L, BLK, GS, TAPS, RANK, TOPK = 2304, 128, 32, 8, 6912, 5, 8, 16, 2, 256, 16
EPS = 1e-6

def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__": continue
            a, b = v["data_offsets"]; f.seek(base + a); raw = f.read(b - a)
            assert v["dtype"] == "BF16", (k, v["dtype"])
            out[k] = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
    return out

def rms(x, w=None, eps=EPS):
    y = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
    return (y * w if w is not None else y).astype(np.float32)

def rope(x, pos, theta=1e4):   # x [rows, heads, 128], neox
    inv = 1.0 / theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :].astype(np.float32), np.sin(ang)[:, None, :].astype(np.float32)
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

def conv(x, coef, base, side):   # x [R, H]; coef [R, 2, TAPS, H/16]; base [2, TAPS, H]
    R = x.shape[0]; g = np.arange(H) // GS
    out = (base[side, 0][None] + coef[:, side, 0][:, g]) * x
    for r in range(1, R):
        out[r] += (base[side, 1] + coef[r, side, 1][g]) * x[r - 1]
    return out

def silu(x): return x / (1 + np.exp(-np.clip(x, -80, 80)))

def cmp(name, a, b, tol=2e-2):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)
    print(f"  {name:28s} rel err {rel:.3e}  max|diff| {np.abs(a-b).max():.3e}  |ref| {np.abs(b).max():.3e}  {'ok' if rel < tol else 'MISMATCH'}")
    return rel < tol

def main():
    W = load(sys.argv[1]); d = sys.argv[2].rstrip("/") + "/"
    pos0, n = [int(x) for x in open(d + "meta.txt").read().split()[1::2]]
    p = int(open(d + "meta2.txt").read().split()[1])
    feat = np.fromfile(d + "feat.bin", np.float32).reshape(n, 4 * H)
    print(f"context rows {n} from pos {pos0}, block at p = {p}")
    ctx = rms(feat @ W["fc.weight"].T, W["hidden_norm.weight"])
    ok = True
    for i in range(L):
        k = (ctx @ W[f"layers.{i}.self_attn.k_proj.weight"].T).reshape(n, NKV, HD)
        k = rope(rms(k, W[f"layers.{i}.self_attn.k_norm.weight"]), np.arange(pos0, pos0 + n)).reshape(n, NKV * HD)
        v = ctx @ W[f"layers.{i}.self_attn.v_proj.weight"].T
        K = np.fromfile(d + f"k{i}.bin", np.float16).reshape(-1, NKV * HD); V = np.fromfile(d + f"v{i}.bin", np.float16).reshape(-1, NKV * HD)
        idx = np.arange(pos0, pos0 + n) % K.shape[0]
        ok &= cmp(f"ctx K layer {i}", K[idx].astype(np.float32), k); ok &= cmp(f"ctx V layer {i}", V[idx].astype(np.float32), v)
    emb = np.fromfile(d + "emb.bin", np.float32).reshape(BLK, H)
    x, res = emb, None
    pos = np.arange(p, p + BLK)
    for i in range(L):
        q = f"layers.{i}."
        if res is None: res, xn = x, rms(x, W[q + "input_layernorm.weight"])
        else: res = res + x; xn = rms(res, W[q + "input_layernorm.weight"])
        coefA = (xn @ W[q + "attention_conv.kernel_projection.weight"].T).reshape(BLK, 2, TAPS, H // GS)
        xc = conv(xn, coefA, W[q + "attention_conv.base_kernel"], 0)
        if i == 0:
            cmp("L0 input layernorm", np.fromfile(d + "s0_xn.bin", np.float32).reshape(BLK, H), xn)
            cmp("L0 conv coefficients", np.fromfile(d + "s0_coefA.bin", np.float32).reshape(BLK, -1), coefA.reshape(BLK, -1))
            cmp("L0 conv in", np.fromfile(d + "s0_xcA.bin", np.float32).reshape(BLK, H), xc)
        qh = (xc @ W[q + "self_attn.q_proj.weight"].T).reshape(BLK, NQ, HD); kh = (xc @ W[q + "self_attn.k_proj.weight"].T).reshape(BLK, NKV, HD)
        vh = (xc @ W[q + "self_attn.v_proj.weight"].T).reshape(BLK, NKV, HD)
        qh = rope(rms(qh, W[q + "self_attn.q_norm.weight"]), pos); kh = rope(rms(kh, W[q + "self_attn.k_norm.weight"]), pos)
        if i == 0:
            e = np.fromfile(d + "s0_qkv.bin", np.float32).reshape(BLK, -1)
            cmp("L0 q (norm+rope)", e[:, :NQ * HD], qh.reshape(BLK, -1)); cmp("L0 k (norm+rope)", e[:, NQ * HD:NQ * HD + NKV * HD], kh.reshape(BLK, -1)); cmp("L0 v", e[:, NQ * HD + NKV * HD:], vh.reshape(BLK, -1))
        # keys: committed context (positions in the window) + the block, non-causal
        Kc = np.fromfile(d + f"k{i}.bin", np.float16).reshape(-1, NKV * HD).astype(np.float32); Vc = np.fromfile(d + f"v{i}.bin", np.float16).reshape(-1, NKV * HD).astype(np.float32)
        cpos = np.arange(max(0, p - 2048), p)   # every committed row below the block (the dump's context call may precede some of them; they are in the ring)
        Kk = np.concatenate([Kc[cpos % Kc.shape[0]], kh.reshape(BLK, -1)]).reshape(-1, NKV, HD); Vv = np.concatenate([Vc[cpos % Vc.shape[0]], vh.reshape(BLK, -1)]).reshape(-1, NKV, HD)
        att = np.zeros((BLK, NQ, HD), np.float32)
        for h in range(NQ):
            g = h // (NQ // NKV); sc = qh[:, h] @ Kk[:, g].T / np.sqrt(HD); sc -= sc.max(-1, keepdims=True); e = np.exp(sc); att[:, h] = (e / e.sum(-1, keepdims=True)) @ Vv[:, g]
        if i == 0: cmp("L0 attention", np.fromfile(d + "s0_att.bin", np.float32).reshape(BLK, -1), att.reshape(BLK, -1))
        o = att.reshape(BLK, -1) @ W[q + "self_attn.o_proj.weight"].T
        if i == 0: cmp("L0 o_proj", np.fromfile(d + "s0_o.bin", np.float32).reshape(BLK, H), o)
        o = conv(o, coefA, W[q + "attention_conv.base_kernel"], 1)
        res = res + o; xn = rms(res, W[q + "post_attention_layernorm.weight"])
        if i == 0: cmp("L0 post-attention norm", np.fromfile(d + "s0_xn2.bin", np.float32).reshape(BLK, H), xn)
        coefM = (xn @ W[q + "mlp_conv.kernel_projection.weight"].T).reshape(BLK, 2, TAPS, H // GS)
        xc = conv(xn, coefM, W[q + "mlp_conv.base_kernel"], 0)
        act = silu(xc @ W[q + "mlp.gate_proj.weight"].T) * (xc @ W[q + "mlp.up_proj.weight"].T)
        if i == 0: cmp("L0 swiglu act", np.fromfile(d + "s0_act.bin", np.float32).reshape(BLK, -1), act)
        m = act @ W[q + "mlp.down_proj.weight"].T
        if i == 0: cmp("L0 down_proj", np.fromfile(d + "s0_down.bin", np.float32).reshape(BLK, H), m)
        x = conv(m, coefM, W[q + "mlp_conv.base_kernel"], 1)
        eng = np.fromfile(d + f"layer{i}.bin", np.float32).reshape(BLK, H)
        ok &= cmp(f"block output layer {i}", eng, x)
    final = rms(res + x, W["norm.weight"])
    eng_final = np.fromfile(d + "final.bin", np.float32).reshape(BLK, H)
    ok &= cmp("final hidden", eng_final, final)
    # candidates: top 16 of the engine's logits rows (from the engine's own final rows), then the greedy walk
    logits = np.fromfile(d + "logits.bin", np.float32).reshape(BLK - 1, -1)
    cand = np.fromfile(d + "cand.bin", np.int32).reshape(BLK - 1, TOPK); cval = np.fromfile(d + "cval.bin", np.float32).reshape(BLK - 1, TOPK)
    ref_ids = np.argsort(-logits, axis=1, kind="stable")[:, :TOPK]
    print("  top-16 ids match (as sets)     :", all(set(cand[l]) == set(ref_ids[l]) for l in range(BLK - 1)), "| top-1 match:", all(cand[l, 0] == ref_ids[l, 0] for l in range(BLK - 1)))
    hp = (eng_final[1:] @ W["candidate_selector.hidden_projection.weight"].T)
    ok &= cmp("hidden projection", np.fromfile(d + "hp.bin", np.float32).reshape(BLK - 1, RANK), hp)
    A, B = W["candidate_selector.predecessor_codebook"], W["candidate_selector.successor_codebook"]
    drafts = np.fromfile(d + "drafts.bin", np.int32); pred = int(drafts[0]); walk = []
    for l in range(BLK - 1):
        s = cval[l] + np.array([((A[pred] * hp[l]) * B[c]).sum() for c in cand[l]]); best = int(np.argmax(s)); pred = int(cand[l, best]); walk.append(pred)
    print("  anchor token", int(drafts[0]), "| engine drafts", drafts[1:].tolist(), "| reference walk", walk, "| match:", drafts[1:].tolist() == walk)
    print("ALL STAGES OK" if ok else "SOME STAGE MISMATCHES (the first one is the bug)")

main()
