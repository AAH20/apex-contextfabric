"""Foundational primitives and data contracts for Apex_ContextFabric.

Sovereign 32x KV-Cache Transform Coding (KVTC), Hierarchical Context Memory Storage (CMX)
& Cross-Agent Attention Deduplication Engine for NVIDIA Blackwell, H100 & Nemotron-3-Ultra.
Zero external dependencies (pure Python 3.9+ standard library only).
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class MemoryTier(str, enum.Enum):
    TIER0_HBM3E = "TIER0_HBM3E"         # Sub-1us latency on GPU High-Bandwidth Memory
    TIER1_HOST_RAM = "TIER1_HOST_RAM"   # ~5us latency on Host RAM (Pinned memory)
    TIER2_NVME_CMX = "TIER2_NVME_CMX"   # ~45us latency on CMX NVMe storage tier
    TIER3_EDGE_CLEF = "TIER3_EDGE_CLEF" # ~100us distributed Cloudflare Clef edge cache


class QuantizationPrecision(str, enum.Enum):
    FP16_BASELINE = "FP16_BASELINE"     # 16-bit uncompressed baseline (2.0 bytes/element)
    FP8_STANDARD = "FP8_STANDARD"       # 8-bit standard quantization (1.0 bytes/element, 2x)
    INT4_TRANSFORM = "INT4_TRANSFORM"   # 4-bit PCA transform-coded (0.5 bytes/element, 32x effective with sparsity)
    FP4_TRANSFORM = "FP4_TRANSFORM"     # 4-bit floating point micro-scaled (0.5 bytes/element)


@dataclass
class KVBlockDescriptor:
    block_id: str
    token_count: int
    head_count: int
    head_dim: int
    precision: QuantizationPrecision
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    tier: MemoryTier
    reference_count: int
    is_attention_sink: bool
    creation_timestamp: float
    sha256_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_saved_bytes(self) -> int:
        return max(0, self.raw_bytes - self.compressed_bytes)


@dataclass
class KVTCMetrics:
    original_memory_mb: float
    compressed_memory_mb: float
    compression_ratio: float
    memory_savings_pct: float
    perplexity_delta: float
    transform_latency_us: float
    outlier_tokens_isolated: int
    sparse_tokens_pruned: int


@dataclass
class CMXTieringStatus:
    tier0_hbm_mb: float
    tier1_ram_mb: float
    tier2_nvme_mb: float
    total_blocks_managed: int
    hbm_hit_rate_pct: float
    hbm_capacity_limit_mb: float
    eviction_latency_us: float
    prefetch_latency_us: float


@dataclass
class CrossAgentMeshSummary:
    active_subagents: int
    total_swarm_tokens: int
    unique_tokens_retained: int
    deduplication_ratio_pct: float
    shared_blocks_count: int
    hbm_reclaimed_gb: float


@dataclass
class ContextFabricState:
    session_id: str
    model_name: str
    context_tokens: int
    kvtc_metrics: KVTCMetrics
    cmx_status: CMXTieringStatus
    mesh_summary: CrossAgentMeshSummary
    ledger_sha256: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
