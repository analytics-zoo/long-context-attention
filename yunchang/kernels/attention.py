from typing import Optional, Tuple
from yunchang.globals import HAS_FLASH_ATTN, HAS_FLASH_ATTN_HOPPER, HAS_FLASHINFER
import math
import torch
if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


if HAS_FLASH_ATTN_HOPPER:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn_interface import flash_attn_func as flash3_attn_func
else:
    flash_attn_forward_hopper = None
    flash_attn_func_hopper_backward = None
    flash3_attn_func = None

if HAS_FLASHINFER:
    from flashinfer.prefill import single_prefill_with_kv_cache
    _LOG2_E = math.log2(math.e)

import torch.nn.functional as F


import torch
aten = torch.ops.aten

naned = False
import torch.nn.functional as F

_original_F_sdpa = F.scaled_dot_product_attention
def chunk_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    chunk_size=1024,
):
    if chunk_size is None or query.size(2) <= chunk_size:
        return _original_F_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )

    if scale is not None:
        return _original_F_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )
    
    if is_causal:
        warnings.warn("Chunked computation may not work correctly with causal attention. "
                      "Consider setting chunk_size=None for causal attention.")
    
    if dropout_p > 0:
        warnings.warn("Dropout is applied independently to each chunk, which may "
                      "result in slightly different behavior compared to non-chunked version.")
    
    Lq = query.size(2)
    query_chunks = torch.split(query, chunk_size, dim=2)
    
    mask_chunks = None
    if attn_mask is not None:
        split_dim = -2 if attn_mask.dim() >= 2 else 0
        if attn_mask.size(split_dim) == 1:
            mask_chunks = [attn_mask] * len(query_chunks)
        elif attn_mask.size(split_dim) == Lq:
            mask_chunks = torch.split(attn_mask, chunk_size, dim=split_dim)
        else:
            raise ValueError(f"Attention mask size {attn_mask.size()} is incompatible "
                             f"with query size {query.size()} for chunked computation")
    else:
        mask_chunks = [None] * len(query_chunks)
    
    output_chunks = []
    
    for q_chunk, m_chunk in zip(query_chunks, mask_chunks):
        chunk_output = F.scaled_dot_product_attention(
            q_chunk, key, value, 
            attn_mask=m_chunk,
            dropout_p=dropout_p,
            is_causal=is_causal
        )
        output_chunks.append(chunk_output)
    
    return torch.cat(output_chunks, dim=2)

def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
    op_type="flash",
):
    assert op_type in ["flash", "efficient"], f"Invalid op_type: {op_type}"
    """
    q shape (bs, seqlen, nhead, hs)
    k shape (bs, seqlen, nhead, hs)
    v shape (bs, seqlen, nhead, hs)
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if op_type == "flash":
        #out, lse = aten._scaled_dot_product_flash_attention(
        #    q,
        #    k,
        #    v,
        #    dropout_p=dropout_p,
        #    is_causal=causal,
        #    scale=softmax_scale,
        #)[:2]

        # dtype = torch.float32
        # out = chunk_scaled_dot_product_attention(
        #    q.contiguous(),
        #    k.contiguous(),
        #    v.contiguous(),
        #    dropout_p=dropout_p,
        #    is_causal=causal,
        #    scale=softmax_scale,
        #    chunk_size=512,
        # )

        import xe_addons
        import math
        head_size = q.shape[-1]
        attn_mask = None
        dtype = torch.float16
        scale = 1 / math.sqrt(head_size)
        out = xe_addons.sdp_non_causal(
            q.contiguous().to(dtype),
            k.contiguous().to(dtype),
            v.contiguous().to(dtype),
            attn_mask,
            scale)
        
        global naned
        def has_nan(x):
            return torch.isnan(x).any()

        if not naned:
            if has_nan(q):
                print("q naned: ", q.device, " ", q.shape)
                naned = True
            if has_nan(k):
                print("k naned: ", k.device, " ", k.shape)
                naned = True
            if has_nan(v):
                print("v naned: ", v.device, " ", v.shape)
                naned = True
            if has_nan(out):
                print("out naned: ", out.device, " ", out.shape)
                naned = True

        out = out.contiguous().to(q.dtype)
        
        lse = torch.ones((q.shape[0], q.shape[1], q.shape[2]),  device=out.device, dtype=q.dtype)
        # torch.xpu.empty_cache()
    elif op_type == "efficient":
        out, lse = aten._scaled_dot_product_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    else:
        raise ValueError(f"Invalid op_type: {op_type}")
    
    out = out.transpose(1, 2)
    lse = lse.to(q.dtype)
    return out, lse

def pytorch_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    block_dq_buffer=None,  # Add new parameters with default values
    block_dk_buffer=None,
    block_dv_buffer=None,
    dropout_p=0.0,
    softmax_scale=None,
    bwd_causal=None,  # This will replace the original causal parameter
    window_size=None,
    softcap=None,
    alibi_slopes=None,
    deterministic=True,
    rng_state=None,
    *args,
    **kwargs,
):
    raise RuntimeError("Not implemented backward for AttnType.TORCH")
    # TODO(optim): use pytorch _scaled_dot_product_efficient_attention_backward
    # Use efficient attention backward
    # https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml#L2874


def flash_attn_forward(q, k, v, 
        dropout_p = 0.0, 
        softmax_scale = None, 
        causal=False, 
        window_size=(-1, -1), 
        softcap=None, 
        alibi_slopes=None, 
        return_softmax=False):
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < '2.6.3':
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p = dropout_p,
            softmax_scale = softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p = dropout_p,
            softmax_scale = softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    return block_out, block_lse

def flash_attn_backward(dout, q, k, v, out, softmax_lse, block_dq_buffer, block_dk_buffer, block_dv_buffer, dropout_p, softmax_scale, 
    bwd_causal, window_size, softcap, alibi_slopes, deterministic, rng_state):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    assert HAS_FLASH_ATTN
    if flash_attn.__version__ < '2.6.3':
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            block_dq_buffer,
            block_dk_buffer,
            block_dv_buffer,
            dropout_p,
            softmax_scale,
            bwd_causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            rng_state,
        )
    else:
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            block_dq_buffer,
            block_dk_buffer,
            block_dv_buffer,
            dropout_p,
            softmax_scale,
            bwd_causal,
            window_size[0],  # Pass window_size_left
            window_size[1],  # Pass window_size_right
            softcap,
            alibi_slopes,
            deterministic,
            rng_state,
        )
    

def flash_attn3_func_forward(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax):
    assert HAS_FLASH_ATTN_HOPPER
    # current signature of flash_attn_forward_hopper:
    # (q, k, v, softmax_scale, causal, window_size, descale_q=None, descale_k=None, descale_v=None, gqa_parallel=False)
    out, q, k, v, out_padded, softmax_lse, S_dmask = flash_attn_forward_hopper(
        q, k, v, softmax_scale, causal, window_size
    )
    return out, softmax_lse

def flash_attn3_func_backward(dout, q, k, v, out, softmax_lse, 
                                    block_dq_buffer, block_dk_buffer, block_dv_buffer, 
                                    dropout_p, softmax_scale, 
                                    bwd_causal, window_size, softcap, alibi_slopes, deterministic, rng_state):
    # (dout, q, k, v, out, softmax_lse, dq, dk, dv, softmax_scale, causal):
    assert HAS_FLASH_ATTN_HOPPER, f"FlashAttention Hopper is not available"

    flash_attn_func_hopper_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        block_dq_buffer,
        block_dk_buffer,
        block_dv_buffer,
        softmax_scale,
        bwd_causal,
        window_size,
        deterministic,
    )

def flashinfer_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: Optional[float] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert HAS_FLASHINFER, "FlashInfer is not available"
    if q.ndim == 4:
        if q.shape[0] >1:
            raise ValueError("batch size > 1 is not supported")
        out, lse = single_prefill_with_kv_cache(
            q[0],
            k[0],
            v[0],
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        lse = lse.transpose(0, 1)
        out, lse = out.unsqueeze(0),lse.unsqueeze(0)
    elif q.ndim == 3:
        out, lse = single_prefill_with_kv_cache(
            q,
            k,
            v,
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        lse = lse.transpose(0, 1)
    else:
        raise ValueError(f"Invalid input shape: {q.shape}")
    lse = lse / _LOG2_E
    return out, lse


def flashinfer_attn_backbward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: Optional[float] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise RuntimeError("Not implemented backward for AttnType.FLASHINFER")
