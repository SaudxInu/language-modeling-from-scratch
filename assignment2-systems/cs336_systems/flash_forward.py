import math
from einops import einsum, rearrange
import torch
import triton
import triton.language as tl


class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        TILE_SIZE_BQ = 16
        TILE_SIZE_BK = 16
        b = Q.shape[0]
        n_q, n_k, n_v = Q.shape[1], K.shape[1], V.shape[1]
        d_q, d_k, d_v = Q.shape[2], K.shape[2], V.shape[2]
        assert n_k == n_v, "K and V must have the same number of tokens."
        assert d_q == d_k == d_v, "Q, K, and V must have the same hidden dimension."
        O = torch.zeros(
            (b, n_q, d_v),
            dtype=Q.dtype,
            device=Q.device,
        )
        L = torch.zeros(
            (b, n_q),
            dtype=Q.dtype,
            device=Q.device,
        )
        for k in range(b):
            for i in range(0, n_q, TILE_SIZE_BQ):
                q_i = Q[k, i : i + TILE_SIZE_BQ, :]
                m_i_j_min_1 = torch.full(
                    (TILE_SIZE_BQ,),
                    fill_value=float("-inf"),
                    dtype=Q.dtype,
                    device=Q.device,
                )
                l_i_j_min_1 = torch.zeros(
                    (TILE_SIZE_BQ,),
                    dtype=Q.dtype,
                    device=Q.device,
                )
                o_i_j_min_1 = torch.zeros(
                    (TILE_SIZE_BQ, d_v),
                    dtype=Q.dtype,
                    device=Q.device,
                )
                for j in range(0, n_k, TILE_SIZE_BK):
                    k_j = K[k, j : j + TILE_SIZE_BK, :]
                    v_j = V[k, j : j + TILE_SIZE_BK, :]
                    s_i_j = einsum(q_i, k_j, "BQ d_k, BK d_k -> BQ BK") / math.sqrt(d_k)
                    m_i_j = torch.maximum(m_i_j_min_1, torch.max(s_i_j, dim=-1).values)
                    p_i_j = torch.exp(s_i_j - rearrange(m_i_j, "BQ -> BQ 1"))
                    correction_i_j = torch.exp(m_i_j_min_1 - m_i_j)
                    l_i_j = correction_i_j * l_i_j_min_1 + p_i_j.sum(dim=-1)
                    o_i_j = einsum(
                        torch.diag(correction_i_j),
                        o_i_j_min_1,
                        "BQ BQ, BQ d_v -> BQ d_v",
                    ) + einsum(p_i_j, v_j, "BQ BK, BK d_v -> BQ d_v")
                    m_i_j_min_1 = m_i_j
                    l_i_j_min_1 = l_i_j
                    o_i_j_min_1 = o_i_j
                O[k, i : i + TILE_SIZE_BQ, :] = einsum(
                    torch.diag(l_i_j_min_1**-1),
                    o_i_j_min_1,
                    "BQ BQ, BQ d_v -> BQ d_v",
                )
                L[k, i : i + TILE_SIZE_BQ] = m_i_j_min_1 + torch.log(l_i_j_min_1)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError(
            "Backward pass is not implemented for PyTorchFlashAttention."
        )


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_i = tl.load(Q_block_ptr)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    m_i_j_min_1 = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i_j_min_1 = tl.full((Q_TILE_SIZE,), 0.0, dtype=tl.float32)
    O_i_j_min_1 = tl.full((Q_TILE_SIZE, D), 0.0, dtype=tl.float32)
    for k in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)

        V_j = tl.load(V_block_ptr)

        S_i_j = (
            tl.dot(
                Q_i,
                K_j,
            )
            / scale
        )

        m_i_j = tl.maximum(m_i_j_min_1, tl.max(S_i_j, axis=-1))

        P_i_j = tl.exp(S_i_j - m_i_j[:, None])

        correction_i_j = tl.exp(m_i_j_min_1 - m_i_j)

        l_i_j = correction_i_j * l_i_j_min_1 + tl.sum(P_i_j, axis=-1)

        O_i_j_min_1 = tl.dot(correction_i_j[:, None], O_i_j_min_1)

        O_i_j_min_1 = tl.dot(P_i_j.to(V_j.dtype), V_j, acc=O_i_j_min_1)

        m_i_j_min_1 = m_i_j

        l_i_j_min_1 = l_i_j

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))

        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr.store(O_i_j_min_1, boundary_check=(0, 1))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0),
    )
    L_block_ptr.store(l_i_j_min_1, boundary_check=(0,))


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        b = Q.shape[0]
        n_q, n_k, n_v = Q.shape[1], K.shape[1], V.shape[1]
        d_q, d_k, d_v = Q.shape[2], K.shape[2], V.shape[2]
        assert n_k == n_v, "K and V must have the same number of tokens."
        assert d_q == d_k == d_v, "Q, K, and V must have the same hidden dimension."
        assert Q.is_contiguous(), "Our pointer arithmetic will assume contiguous Q."
        assert K.is_contiguous(), "Our pointer arithmetic will assume contiguous K."
        assert V.is_contiguous(), "Our pointer arithmetic will assume contiguous V."
        O = torch.zeros(
            (b, n_q, d_v),
            dtype=Q.dtype,
            device=Q.device,
        )
        L = torch.zeros(
            (b, n_q),
            dtype=Q.dtype,
            device=Q.device,
        )
        flash_fwd_kernel[(tl.cdiv(n_q, ctx.Q_TILE_SIZE), b)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            Q.shape[1],
            K.shape[1],
            1 / math.sqrt(d_k),
            D=d_q,
            Q_TILE_SIZE=16,
            K_TILE_SIZE=16,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
