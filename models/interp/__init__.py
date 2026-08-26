"""Mechanistic-interpretability access layer for the model roster."""

from .base import CaptureSpec, InterpModel, RandomWeightSpec
from .conventions import StreamView, Task, TokenLayout
from .hooks import ForwardCapture, ForwardPatch, FunctionObserver, PatchSpec
from .trace import AttentionRecord, Trace

__all__ = [
    "CaptureSpec",
    "InterpModel",
    "RandomWeightSpec",
    "StreamView",
    "Task",
    "TokenLayout",
    "ForwardCapture",
    "ForwardPatch",
    "FunctionObserver",
    "PatchSpec",
    "AttentionRecord",
    "Trace",
]
