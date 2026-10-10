"""Core package for Apex_ContextFabric."""

from apex_contextfabric.core.primitives import (
    CMXTieringStatus,
    ContextFabricState,
    CrossAgentMeshSummary,
    KVBlockDescriptor,
    KVTCMetrics,
    MemoryTier,
    QuantizationPrecision,
)
from apex_contextfabric.core.kvtc_transform_coder import KVTCTransformCoder
from apex_contextfabric.core.cmx_hierarchical_tier import CMXHierarchicalTierManager
from apex_contextfabric.core.cross_agent_mesh_dedupe import CrossAgentMeshDeduplicator
from apex_contextfabric.core.attention_sink_pruner import AttentionSinkPruner

__all__ = [
    "CMXTieringStatus",
    "ContextFabricState",
    "CrossAgentMeshSummary",
    "KVBlockDescriptor",
    "KVTCMetrics",
    "MemoryTier",
    "QuantizationPrecision",
    "KVTCTransformCoder",
    "CMXHierarchicalTierManager",
    "CrossAgentMeshDeduplicator",
    "AttentionSinkPruner",
]
