"""Package entry point.

Upstream imports LLM eagerly, which pulls torch -> triton -> flash-attn. Those CUDA
wheels do not exist on a Windows CPU box, so pure-Python tests cannot even be collected.
Resolve LLM lazily; `from nanovllm import LLM` behaves identically on GPU machines.
Local adaptation only, not part of upstream 1.1-done / 2.1-done.
"""
from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
