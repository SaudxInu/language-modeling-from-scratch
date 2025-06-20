import torch
import triton
import triton.language as tl

from flash_forward import TritonFlashAttention
from tests.test_attention import _attention_and_lse


configs = []
for mode in ["forward", "backward"]:
    for precision in [torch.float32, torch.bfloat16]:
        for d in [16, 32, 64]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["sequence_length"],
                    x_vals=[2**i for i in range(7, 15, 1)],
                    x_log=True,
                    line_arg="provider",
                    line_vals=["triton", "torch"],
                    line_names=["Triton", "Torch"],
                    styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
                    ylabel="s",
                    plot_name="flash attention",
                    args={"mode": mode, "d": d, "precision": precision},
                )
            )


@triton.testing.perf_report(configs)
def benchmark(sequence_length, provider, mode=mode, d=d, precision=precision):
    Q = torch.randn(
        (1, sequence_length, d), device="cuda", dtype=precision, requires_grad=True
    )
    K = torch.randn(
        (1, sequence_length, d), device="cuda", dtype=precision, requires_grad=True
    )
    V = torch.randn(
        (1, sequence_length, d), device="cuda", dtype=precision, requires_grad=True
    )
    quantiles = [0.5, 0.2, 0.8]
    if mode == "forward":
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _attention_and_lse(Q, K, V), quantiles=quantiles
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: TritonFlashAttention.apply(Q, K, V), quantiles=quantiles
            )
    if mode == "backward":
        if provider == "torch":
            y = _attention_and_lse(Q, K, V)[0]
            dy = torch.randn_like(y, dtype=torch.float32)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: y.backward(dy, retain_graph=True), quantiles=quantiles
            )
        if provider == "triton":
            y = TritonFlashAttention.apply(Q, K, V)
            dy = torch.randn_like(y, dtype=torch.float32)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: y.backward(dy, retain_graph=True), quantiles=quantiles
            )
    return ms, min_ms, max_ms


# benchmark.run(show_plots=True, print_data=True)


def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3,
        n_heads,
        sequence_length,
        d_head,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    # flash = torch.compile(TritonFlashAttention.apply)
    flash = torch.compile(_attention_and_lse)

    def flash_forward_backward():
        o = flash(q, k, v, True)[0]
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=100, warmup=10)

    print(results)


# test_timing_flash_forward_backward()
