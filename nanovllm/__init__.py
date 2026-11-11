"""Local adaptation for GPU-less machines.

Upstream imports LLM eagerly, which drags in torch -> triton -> flash-attn when
anything under `nanovllm` is imported (e.g. `nanovllm.engine.stats`). Those wheels
do not exist on Windows/CPU. Keeping the two public names lazy changes no public
API: `from nanovllm import LLM, SamplingParams` still works on the GPU box, while
the pure-Python 1.1 tests can be collected locally.
"""


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LLM", "SamplingParams"]
