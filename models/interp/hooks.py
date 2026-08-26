"""Model-agnostic hook primitives: capture, patch, and function observation.

Three deliberate design rules, each one a direct response to a way the previous
implementation went wrong:

1. **Modules are resolved at run time, never cached across ``fit()``.**
   All three estimators rebuild ``self.model_`` inside ``fit()``.  A handle
   taken before ``fit`` silently points at a dead module and its hook never
   fires -- producing an empty capture that looks like a legitimate result.
   Every primitive here takes a zero-argument *resolver* and calls it inside
   ``__enter__``, i.e. after ``fit`` and immediately before the forward pass.

2. **Every forward call is recorded; nothing is overwritten.**
   Modules run more than once per forward pass (TabPFN's item attention runs
   twice per block; every model chunks long sequences).  Keeping "the last
   call" silently returns a different object than the caller thinks.  We keep a
   list and let the adapter assert an expected call count.

3. **Nothing is monkeypatched permanently.**
   ``FunctionObserver`` swaps a module-level function only for the duration of
   a ``with`` block, always delegates the real computation to the original, and
   restores in a ``finally``.  It cannot change what the model computes.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn

Resolver = Callable[[], nn.Module]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    """One forward call of one hooked module."""

    name: str
    call_index: int
    output: torch.Tensor | tuple

    @property
    def shape(self) -> tuple[int, ...] | None:
        return tuple(self.output.shape) if torch.is_tensor(self.output) else None


class ForwardCapture:
    """Record the output of named modules for every forward call.

    ``resolvers`` maps a name to a callable returning the module.  The callable
    is invoked in ``__enter__``, so it always sees the live post-``fit`` module.

    Outputs are detached and moved to CPU by default so a long run does not pin
    the whole activation history on the GPU.
    """

    def __init__(
        self,
        resolvers: dict[str, Resolver],
        *,
        to_cpu: bool = True,
        select: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
        self._resolvers = resolvers
        self._to_cpu = to_cpu
        self._select = select or (lambda out: out)
        self.records: dict[str, list[CallRecord]] = {name: [] for name in resolvers}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ForwardCapture":
        for name, resolve in self._resolvers.items():
            module = resolve()
            if module is None:
                raise RuntimeError(f"ForwardCapture: resolver for {name!r} returned None")
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = self._select(output)
            if torch.is_tensor(tensor):
                tensor = tensor.detach()
                if self._to_cpu:
                    tensor = tensor.cpu()
            self.records[name].append(
                CallRecord(name=name, call_index=len(self.records[name]), output=tensor)
            )

        return hook

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- accessors -----------------------------------------------------------

    def call_counts(self) -> dict[str, int]:
        return {name: len(recs) for name, recs in self.records.items()}

    def require_call_count(self, expected: int, *, where: str = "") -> None:
        """Assert every hooked module fired exactly ``expected`` times.

        This is the guard against silent chunking: if the model splits the
        forward pass, the count goes up and we refuse to return a trace built
        from one arbitrary chunk.
        """
        bad = {n: c for n, c in self.call_counts().items() if c != expected}
        if bad:
            raise AssertionError(
                f"{where or 'ForwardCapture'}: expected {expected} forward call(s) per "
                f"hooked module, got {bad}. Either the model chunked the forward pass "
                "(reduce n, or add chunk reassembly) or the hook is misplaced."
            )

    def only(self, name: str) -> torch.Tensor:
        """The single recorded output for ``name``; raises if not exactly one."""
        recs = self.records[name]
        if len(recs) != 1:
            raise AssertionError(
                f"ForwardCapture.only({name!r}): expected 1 call, got {len(recs)} "
                f"with shapes {[r.shape for r in recs]}"
            )
        return recs[0].output


# ---------------------------------------------------------------------------
# Patch (the write path)
# ---------------------------------------------------------------------------


@dataclass
class PatchSpec:
    """Replace part of a module's output on a chosen forward call.

    ``positions`` selects along the sequence axis of the canonical stream.
    ``None`` means "the whole sequence".  ``values`` must broadcast to the
    selected region.

    ``call_index`` picks which forward call to act on when a module runs more
    than once; ``None`` means every call (only safe when the module runs once).
    """

    module: str
    values: torch.Tensor
    positions: Sequence[int] | slice | None = None
    call_index: int | None = 0
    #: Set by the adapter: writes ``values`` into ``output`` honouring the
    #: model's native tensor rank (per-cell vs per-row).
    writer: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor] | None = None


class ForwardPatch:
    """Overwrite hooked module outputs according to ``PatchSpec``s.

    Returning a value from a ``forward_hook`` replaces the module's output, so
    the patched state genuinely flows into the next block -- this is a causal
    intervention, not an observation.
    """

    def __init__(self, resolvers: dict[str, Resolver], specs: Iterable[PatchSpec]) -> None:
        self._resolvers = resolvers
        self._specs: dict[str, list[PatchSpec]] = {}
        for spec in specs:
            if spec.module not in resolvers:
                raise KeyError(
                    f"PatchSpec targets {spec.module!r} which is not a hookable module. "
                    f"Known: {sorted(resolvers)}"
                )
            self._specs.setdefault(spec.module, []).append(spec)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._counters: dict[str, int] = {}
        self.applied: dict[str, int] = {}

    def __enter__(self) -> "ForwardPatch":
        for name in self._specs:
            module = self._resolvers[name]()
            self._counters[name] = 0
            self.applied[name] = 0
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            call_index = self._counters[name]
            self._counters[name] += 1
            new_output = output
            for spec in self._specs[name]:
                if spec.call_index is not None and spec.call_index != call_index:
                    continue
                if spec.writer is None:
                    raise RuntimeError(
                        f"PatchSpec for {name!r} has no writer; the adapter must "
                        "attach one so the write honours the model's tensor rank."
                    )
                new_output = spec.writer(new_output, spec.values, spec.positions)
                self.applied[name] += 1
            return new_output

        return hook

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def require_applied(self, expected: int) -> None:
        total = sum(self.applied.values())
        if total != expected:
            raise AssertionError(
                f"ForwardPatch: expected {expected} patch application(s), applied {total} "
                f"({self.applied}). The target module never ran, or call_index is wrong."
            )


# ---------------------------------------------------------------------------
# Function observation (attention probabilities without changing the model)
# ---------------------------------------------------------------------------


@dataclass
class ObservedCall:
    """One observed call of a wrapped function."""

    index: int
    payload: dict[str, Any] = field(default_factory=dict)


class FunctionObserver:
    """Temporarily wrap a module-level function with a delegating observer.

    The wrapper calls the ORIGINAL function for the returned value, so the
    model's numerics are untouched; ``observe`` is handed the call's arguments
    and the original's output purely for recording.  The original is restored
    in ``__exit__`` even if the body raises.

    This is how attention probabilities are recovered from models that use
    fused SDPA/FlashAttention and never materialise them: we recompute the
    probabilities from the same q/k the real kernel saw, and (in the adapters)
    assert that ``probs @ v`` reproduces the real output.
    """

    def __init__(
        self,
        owner: Any,
        attribute: str,
        observe: Callable[[tuple, dict, Any, int], dict[str, Any] | None],
        *,
        staticmethod_: bool = False,
    ) -> None:
        self._owner = owner
        self._attribute = attribute
        self._observe = observe
        self._staticmethod = staticmethod_
        self._original: Any = None
        self.calls: list[ObservedCall] = []
        self.enabled = True

    def __enter__(self) -> "FunctionObserver":
        self._original = getattr(self._owner, self._attribute)
        original = self._original
        # For staticmethods accessed off a class, getattr already unwraps to a
        # plain function, so calling it directly is correct.

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            if self.enabled:
                index = len(self.calls)
                payload = self._observe(args, kwargs, out, index)
                if payload is not None:
                    self.calls.append(ObservedCall(index=index, payload=payload))
            return out

        setattr(self._owner, self._attribute, staticmethod(wrapper) if self._staticmethod else wrapper)
        return self

    def __exit__(self, *exc) -> None:
        setattr(self._owner, self._attribute, self._original)
        self._original = None

    @staticmethod
    def is_clean(owner: Any, attribute: str, expected: Any) -> bool:
        """True when ``owner.attribute`` is still the pristine function."""
        return getattr(owner, attribute) is expected


def stacked(*context_managers) -> ExitStack:
    """Enter several optional context managers, skipping ``None``."""
    stack = ExitStack()
    for cm in context_managers:
        if cm is not None:
            stack.enter_context(cm)
    return stack
