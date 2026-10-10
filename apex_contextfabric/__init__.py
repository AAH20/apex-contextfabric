"""Apex_ContextFabric package.

Sovereign 32x KV-Cache Transform Coding (KVTC), Hierarchical Context Memory Storage (CMX)
& Cross-Agent Attention Deduplication Engine for NVIDIA Blackwell, H100 & Nemotron-3-Ultra.
Zero external dependencies (pure Python 3.9+ standard library only).
"""

__version__ = "1.0.0"
__author__ = "Ahmed Hassan"
__license__ = "Apache-2.0"

from apex_contextfabric.orchestrator import ApexContextFabricOrchestrator
from apex_contextfabric.core.primitives import (
    CMXTieringStatus,
    ContextFabricState,
    CrossAgentMeshSummary,
    KVBlockDescriptor,
    KVTCMetrics,
    MemoryTier,
    QuantizationPrecision,
)

__all__ = [
    "__version__",
    "ApexContextFabricOrchestrator",
    "CMXTieringStatus",
    "ContextFabricState",
    "CrossAgentMeshSummary",
    "KVBlockDescriptor",
    "KVTCMetrics",
    "MemoryTier",
    "QuantizationPrecision",
]
