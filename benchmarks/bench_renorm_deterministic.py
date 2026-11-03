"""Cost of ``is_deterministic=True`` for top-k / top-p renorm (flashinfer-ai/flashinfer#7918)."""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def require_cupti():
    from importlib.metadata import version

    from cupti import cupti  # noqa: F401

    assert int(version("cupti-python").split(".")[0]) >= 13, version("cupti-python")


def median_us(fn, args):
    times = bench_gpu_time(
        fn,
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        enable_cupti=True,
        cold_l2_cache=True,
        input_args=args,
    )
    return float(np.median(times)) * 1e3


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, nargs="+", default=[32000, 128256, 152064])
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 8, 32, 128, 512, 1024]
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[50, 1000])
    parser.add_argument("--top-p", type=float, nargs="+", default=[0.5, 0.9])
    args = parser.parse_args()
    require_cupti()
    torch.manual_seed(42)
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    print(f"flashinfer: {flashinfer.__file__}")
    print(
        f"gpu: {props.name} sm_{props.major}{props.minor} SMs={props.multi_processor_count}"
    )
    print(f"torch: {torch.__version__} cuda: {torch.version.cuda}")
    print("op,param,vocab,batch,default_us,deterministic_us,ratio")
    rows = []
    for vocab in args.vocab:
        for batch in args.batch:
            logits = torch.randn(batch, vocab, device="cuda") * 3.0
            probs = torch.softmax(logits, dim=-1)
            cases = [
                (
                    "top_k_renorm_probs",
                    f"k={k}",
                    flashinfer.sampling.top_k_renorm_probs,
                    torch.full((batch,), k, dtype=torch.int32, device="cuda"),
                )
                for k in args.top_k
            ] + [
                (
                    "top_p_renorm_probs",
                    f"p={p}",
                    flashinfer.sampling.top_p_renorm_probs,
                    torch.full((batch,), p, dtype=torch.float32, device="cuda"),
                )
                for p in args.top_p
            ]
            for op, param, fn, thresh in cases:
                us = {}
                for det in (False, True):
                    us[det] = median_us(
                        lambda probs, thresh, det=det: fn(
                            probs, thresh, is_deterministic=det
                        ),
                        (probs, thresh),
                    )
                ratio = us[True] / us[False]
                rows.append((op, param, vocab, batch, us[False], us[True], ratio))
                print(
                    f"{op},{param},{vocab},{batch},{us[False]:.2f},{us[True]:.2f},{ratio:.3f}"
                )
            del logits, probs
    print("\nsummary (deterministic / default, median us):")
    for op in ("top_k_renorm_probs", "top_p_renorm_probs"):
        r = [x[6] for x in rows if x[0] == op]
        print(
            f"  {op}: ratio min {min(r):.3f} median {np.median(r):.3f} max {max(r):.3f}"
        )


if __name__ == "__main__":
    main()
