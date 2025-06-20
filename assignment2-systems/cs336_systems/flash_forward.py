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
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors

        S = einsum(Q, K, "... BQ d, ... BK d -> ... BQ BK") / math.sqrt(Q.shape[2])

        if ctx.is_causal:
            S = torch.where(
                torch.arange(Q.shape[-2], device=S.device)[None, :, None]
                >= torch.arange(K.shape[-2], device=S.device)[None, None, :],
                S,
                -1e6,
            )

        P = torch.exp(S - L.unsqueeze(-1))

        grad_V = einsum(P, grad_out, "... BQ BK, ... BQ d -> ... BK d")

        grad_P = einsum(grad_out, V, "... BQ d, ... BK d -> ... BQ BK")

        D = (O * grad_out).sum(dim=-1)

        grad_S = P * (grad_P - D.unsqueeze(-1))

        grad_Q = einsum(grad_S, K, "... BQ BK, ... BK d -> ... BQ d") / math.sqrt(
            Q.shape[2]
        )

        grad_K = einsum(grad_S, Q, "... BQ BK, ... BQ d -> ... BK d") / math.sqrt(
            Q.shape[2]
        )

        return grad_Q, grad_K, grad_V, None


# * Tuning kernel is important.
# * No if else in kernels.
# * Keep memory access patterns in mind while tiling.
# * Locks and compare it with re-computation.
# * Tiling and recompuation is used to reduce peak memory usage and save I/O bandwidth.
# * Check the regime for a operation. We want compute bound regime.
# * If in memory bound regime, try to remove redundant read and writes and do as much computation per load.
# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 "Q_TILE_SIZE": q_tile_size,
#                 "K_TILE_SIZE": k_tile_size,
#             },
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for q_tile_size in [64, 128, 256]
#         for k_tile_size in [64, 128, 256]
#         for num_stages in [2, 3, 4]
#         for num_warps in [2, 4, 8]
#     ],
#     key=["N_QUERIES", "N_KEYS"],
# )
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
    is_causal: tl.constexpr,
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
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    m_i_j_min_1 = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_i_j_min_1 = tl.full((Q_TILE_SIZE,), 0.0, dtype=tl.float32)
    O_i_j_min_1 = tl.full((Q_TILE_SIZE, D), 0.0, dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)

        V_j = tl.load(V_block_ptr)

        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale

        if is_causal:
            mask = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[
                :, None
            ] < (j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))[None, :]

            S_i_j = tl.where(mask == 0, S_i_j, S_i_j - 1e6)

        m_i_j = tl.maximum(m_i_j_min_1, tl.max(S_i_j, axis=-1))

        P_i_j = tl.exp(S_i_j - m_i_j[:, None])

        correction_i_j = tl.exp(m_i_j_min_1 - m_i_j)

        l_i_j = correction_i_j * l_i_j_min_1 + tl.sum(P_i_j, axis=-1)

        offs_i = tl.arange(0, Q_TILE_SIZE)
        offs_j = tl.arange(0, Q_TILE_SIZE)
        diag_mask = offs_i[:, None] == offs_j[None, :]

        O_i_j_min_1 = tl.dot(
            diag_mask.to(tl.float32) * correction_i_j[:, None], O_i_j_min_1
        )

        O_i_j = tl.dot(P_i_j.to(V_j.dtype), V_j, acc=O_i_j_min_1)

        m_i_j_min_1 = m_i_j

        l_i_j_min_1 = l_i_j

        O_i_j_min_1 = O_i_j

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))

        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    offs_i = tl.arange(0, Q_TILE_SIZE)
    offs_j = tl.arange(0, Q_TILE_SIZE)
    diag_mask = offs_i[:, None] == offs_j[None, :]

    O_i = tl.dot(diag_mask.to(tl.float32) * (1 / l_i_j_min_1[:, None]), O_i_j_min_1)

    l_i = m_i_j_min_1 + tl.log(l_i_j_min_1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr.store(O_i.to(Q_i.dtype), boundary_check=(0, 1))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    L_block_ptr.store(l_i.to(Q_i.dtype), boundary_check=(0,))


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 "Q_TILE_SIZE": q_tile_size,
#                 "K_TILE_SIZE": k_tile_size,
#             },
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for q_tile_size in [64, 128, 256]
#         for k_tile_size in [64, 128, 256]
#         for num_stages in [2, 3, 4]
#         for num_warps in [2, 4, 8]
#     ],
#     key=["N_QUERIES", "N_KEYS"],
# )
@triton.jit
def flash_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    D_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    lock_ptr,
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
    stride_db,
    stride_dq,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    stride_lockb,
    stride_lockq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    K_j = tl.load(K_block_ptr)

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_j = tl.load(V_block_ptr)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dK_j = tl.full((K_TILE_SIZE, D), 0.0, dtype=tl.float32)
    dV_j = tl.full((K_TILE_SIZE, D), 0.0, dtype=tl.float32)
    # for i in tl.range(tl.cdiv(N_QUERIES, Q_TILE_SIZE), warp_specialize=True):
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr)

        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale

        if is_causal:
            mask = (i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[:, None] < (
                key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            )[None, :]

            S_i_j = tl.where(mask == 0, S_i_j, S_i_j - 1e6)

        L_i = tl.load(L_block_ptr)

        P_i_j = tl.exp(S_i_j - L_i[:, None]).to(tl.float32)

        O_i = tl.load(O_block_ptr)

        dO_i = tl.load(dO_block_ptr)

        dV_j = tl.dot(tl.trans(P_i_j), dO_i.to(tl.float32), dV_j)

        dP_i_j = tl.dot(dO_i.to(tl.float32), tl.trans(V_j).to(tl.float32))

        D_i = tl.load(D_block_ptr).to(tl.float32)

        dS_i_j = P_i_j * (dP_i_j - D_i[:, None]) * scale

        dK_j = tl.dot(tl.trans(dS_i_j), Q_i.to(tl.float32), dK_j)

        lock_ptr_i = lock_ptr + batch_index * stride_lockb + i * stride_lockq

        while tl.atomic_cas(lock_ptr_i, 0, 1) == 1:
            pass

        dQ_block_ptr = tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D),
            strides=(stride_dqq, stride_dqd),
            offsets=(i * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        dQ_i = tl.load(dQ_block_ptr)

        dQ_i = tl.dot(dS_i_j, K_j.to(tl.float32), dQ_i)

        dQ_block_ptr.store(dQ_i, boundary_check=(0, 1))

        tl.debug_barrier()

        tl.atomic_xchg(lock_ptr_i, 0)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr.store(dK_j, boundary_check=(0, 1))

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr.store(dV_j, boundary_check=(0, 1))


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
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
        grid = lambda META: (triton.cdiv(n_q, META["Q_TILE_SIZE"]), b)
        flash_fwd_kernel[grid](
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
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        Q, K, V, O, L = ctx.saved_tensors
        b = Q.shape[0]
        n_q, n_k, n_v = Q.shape[1], K.shape[1], V.shape[1]
        d_q, d_k, d_v = Q.shape[2], K.shape[2], V.shape[2]
        D = (O.to(torch.float32) * grad_out).sum(dim=-1)
        dQ = torch.zeros(
            (b, n_q, d_q),
            dtype=torch.float32,
            device=Q.device,
        )
        dK = torch.zeros(
            (b, n_k, d_k),
            dtype=torch.float32,
            device=Q.device,
        )
        dV = torch.zeros(
            (b, n_k, d_v),
            dtype=torch.float32,
            device=Q.device,
        )
        lock = torch.zeros(
            (b, n_q),
            dtype=torch.int32,
            device=Q.device,
        )
        grid = lambda META: (triton.cdiv(n_k, META["K_TILE_SIZE"]), b)
        flash_bwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            D,
            grad_out,
            dQ,
            dK,
            dV,
            lock,
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
            D.stride(0),
            D.stride(1),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            lock.stride(0),
            lock.stride(1),
            Q.shape[1],
            K.shape[1],
            1 / math.sqrt(d_k),
            D=d_q,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )
        return dQ, dK, dV, None

    # @staticmethod
    # def backward(ctx, grad_out):
    #     Q, K, V, O, L = ctx.saved_tensors

    #     S = einsum(Q, K, "... BQ d, ... BK d -> ... BQ BK") / math.sqrt(Q.shape[2])

    #     if ctx.is_causal:
    #         S = torch.where(
    #             torch.arange(Q.shape[-2], device=S.device)[None, :, None]
    #             >= torch.arange(K.shape[-2], device=S.device)[None, None, :],
    #             S,
    #             -1e6,
    #         )

    #     P = torch.exp(S - L.unsqueeze(-1))

    #     grad_V = einsum(P, grad_out, "... BQ BK, ... BQ d -> ... BK d")

    #     grad_P = einsum(grad_out, V, "... BQ d, ... BK d -> ... BQ BK")

    #     D = (O * grad_out).sum(dim=-1)

    #     grad_S = P * (grad_P - D.unsqueeze(-1))

    #     grad_Q = einsum(grad_S, K, "... BQ BK, ... BK d -> ... BQ d") / math.sqrt(
    #         Q.shape[2]
    #     )

    #     grad_K = einsum(grad_S, Q, "... BQ BK, ... BQ d -> ... BK d") / math.sqrt(
    #         Q.shape[2]
    #     )

    #     return grad_Q, grad_K, grad_V, None
