import argparse
import json
import os
from random import Random
from time import perf_counter

from bench.metrics import format_report, summarize_requests, summarize_steps
from bench.trace import TraceWriter
from bench.workloads import SCENARIOS


def drive(llm, requests, seed: int = 0) -> float:
    """Feed requests by arrival_step, step the engine until all are done. Returns wall seconds.

    TODO(任务 B): 按 arrival_step 塞请求（虚拟时间，不按墙钟）；引擎空闲但还有请求
    没到时只推进步数；每步包一个 NVTX range `step N`；返回测量段的墙钟时长。
    """
    raise NotImplementedError("任务 B：bench.run.drive")


def warmup(llm, requests, n: int = 64):
    """Run a slice of the real workload so torch.compile has seen its shapes before we measure.
    A single short request is not enough: the first big prefill would recompile inside the measurement."""
    from nanovllm import SamplingParams
    rng = Random(-1)
    prompts = [[rng.randint(0, 10000) for _ in range(r.prompt_len)] for r in requests[:n]]
    sp = SamplingParams(max_tokens=4, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)
    # batch size 1 is specialized by torch.compile and needs its own compile (e.g. a lone re-prefill after preemption)
    llm.generate(prompts[:1], sp, use_tqdm=False)
    llm.step_records.clear()
    llm.request_records.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="steady_decode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace", help="write per-step JSONL here")
    p.add_argument("--json", help="write the summary here")
    args = p.parse_args()

    from nanovllm import LLM
    llm = LLM(os.path.expanduser(args.model), enforce_eager=False, max_model_len=4096)
    requests = SCENARIOS[args.scenario](args.seed)
    warmup(llm, requests)

    writer = None
    if args.trace:
        os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
        writer = llm.trace = TraceWriter(args.trace)
    duration = drive(llm, requests, args.seed)
    if writer:
        writer.close()

    steps = summarize_steps(llm.step_records)
    requests = summarize_requests(llm.request_records, duration)
    print(format_report(args.scenario, steps, requests))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"scenario": args.scenario, "steps": steps, "requests": requests}, f, indent=2)


if __name__ == "__main__":
    main()
