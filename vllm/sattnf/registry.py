"""
Simple registries for the different pluggable parts of SAttnF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, Mapping, MutableMapping, TypeVar

from .executor import BlockExecutor
from .hooks import SchedulerHook
from .kv import KVManager
from .mask import MaskStrategy

T = TypeVar("T")


class ComponentNotFoundError(RuntimeError):
    pass


class ComponentRegistry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._registry: Dict[str, type[T]] = {}

    def register(self, name: str, cls: type[T]) -> None:
        key = name.lower()
        if key in self._registry:
            raise ValueError(f"{self.kind} '{name}' already registered.")
        self._registry[key] = cls

    def get(self, name: str) -> type[T]:
        key = name.lower()
        try:
            return self._registry[key]
        except KeyError as exc:
            raise ComponentNotFoundError(
                f"Unknown {self.kind} '{name}'. Available: {list(self._registry)}"
            ) from exc

    def create(self, name: str, **cfg: Any) -> T:
        cls = self.get(name)
        instance = cls()  # type: ignore[call-arg]
        configure = getattr(instance, "configure", None)
        if callable(configure):
            configure(**cfg)
        return instance

    def items(self) -> Iterable[tuple[str, type[T]]]:
        return self._registry.items()


mask_registry: ComponentRegistry[MaskStrategy] = ComponentRegistry("MaskStrategy")
kv_registry: ComponentRegistry[KVManager] = ComponentRegistry("KVManager")
executor_registry: ComponentRegistry[BlockExecutor] = ComponentRegistry("BlockExecutor")
hook_registry: ComponentRegistry[SchedulerHook] = ComponentRegistry("SchedulerHook")


def register_mask(name: str) -> Callable[[type[MaskStrategy]], type[MaskStrategy]]:
    def decorator(cls: type[MaskStrategy]) -> type[MaskStrategy]:
        mask_registry.register(name, cls)
        return cls

    return decorator


def register_kv_manager(name: str) -> Callable[[type[KVManager]], type[KVManager]]:
    def decorator(cls: type[KVManager]) -> type[KVManager]:
        kv_registry.register(name, cls)
        return cls

    return decorator


def register_block_executor(
    name: str,
) -> Callable[[type[BlockExecutor]], type[BlockExecutor]]:
    def decorator(cls: type[BlockExecutor]) -> type[BlockExecutor]:
        executor_registry.register(name, cls)
        return cls

    return decorator


def register_hook(name: str) -> Callable[[type[SchedulerHook]], type[SchedulerHook]]:
    def decorator(cls: type[SchedulerHook]) -> type[SchedulerHook]:
        hook_registry.register(name, cls)
        return cls

    return decorator
