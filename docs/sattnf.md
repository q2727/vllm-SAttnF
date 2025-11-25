# Sparse Attention Framework (SAttnF)

SAttnF is an experiment-friendly abstraction layer that lets researchers plug
different sparse attention algorithms into vLLM without modifying the scheduler
or attention kernels directly.  The framework is fully optional and only
constructed when a `SparseFrameworkConfig` is supplied through `VllmConfig`.

## Components

| Layer | Description | Key API |
| ----- | ----------- | ------- |
| Mask strategy | Selects which blocks/pages (or tokens) should participate in attention for both prefill and decode. | `vllm.sattnf.mask.MaskStrategy` returns a `BlockMask`. |
| KV manager | Owns the lifecycle of KV cache blocks, handles eviction/offload and exposes fetch helpers that mask strategies can call. | `vllm.sattnf.kv.KVManager` |
| Block executor | Runs the actual sparse kernel once KV blocks / token subsets are known. | `vllm.sattnf.executor.BlockExecutor` |
| Scheduler hooks | Observes scheduler events (prefill start, decode loops, request completion) to prefetch blocks or update statistics. | `vllm.sattnf.hooks.SchedulerHook` |

All building blocks are registered via decorators in `vllm.sattnf.registry`.

## Configuration

```python
from vllm.config import SparseFrameworkConfig, VllmConfig

sattnf_cfg = SparseFrameworkConfig(
    enabled=True,
    mask_strategy="streaming_llm",
    mask_strategy_config={"window_size": 512},
    kv_manager="default_paged",
    block_executor="triton_block_sparse",
    scheduler_hooks=["prefetch", "metrics"],
    scheduler_hook_configs={
        "prefetch": {"lookahead": 4},
        "metrics": {"report_interval": 50},
    },
)

vllm_cfg = VllmConfig(
    model_config=...,
    cache_config=...,
    sattnf_config=sattnf_cfg,
)
```

During engine initialisation call `SparseFrameworkManager(vllm_cfg.sattnf_config)`
to instantiate the configured components and obtain handles that can be injected
into the model runner or scheduler.
