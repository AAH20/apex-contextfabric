"""Cross-Agent Semantic Attention Mesh Deduplication Engine.

Enforces zero-redundancy KV-cache sharing across collaborative multi-agent swarms.
When multiple subagents operate over shared documentation, system prompts, or codebase ASTs,
this engine unifies redundant token slices into shared reference-counted blocks.
Zero external dependencies (pure Python 3.9+ standard library only).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from apex_contextfabric.core.primitives import (
    CrossAgentMeshSummary,
    KVBlockDescriptor,
)


class CrossAgentMeshDeduplicator:
    """Manages cross-agent shared attention blocks and reference counting."""

    def __init__(self):
        # Maps content_hash -> KVBlockDescriptor
        self._shared_registry: Dict[str, KVBlockDescriptor] = {}
        # Maps agent_id -> Set of block_ids
        self._agent_allocations: Dict[str, Set[str]] = {}

    def register_agent_block(
        self,
        agent_id: str,
        block_content_hash: str,
        candidate_block: KVBlockDescriptor,
    ) -> Tuple[KVBlockDescriptor, bool]:
        """Registers a KV block for an agent.
        
        Returns:
            (effective_block, is_deduplicated)
            If the block was already registered by another agent, returns the existing
            block with incremented reference_count and is_deduplicated=True.
        """
        if agent_id not in self._agent_allocations:
            self._agent_allocations[agent_id] = set()

        if block_content_hash in self._shared_registry:
            existing_block = self._shared_registry[block_content_hash]
            existing_block.reference_count += 1
            self._agent_allocations[agent_id].add(existing_block.block_id)
            return existing_block, True

        # First time seeing this content
        candidate_block.reference_count = 1
        self._shared_registry[block_content_hash] = candidate_block
        self._agent_allocations[agent_id].add(candidate_block.block_id)
        return candidate_block, False

    def release_agent_context(self, agent_id: str) -> int:
        """Frees all blocks allocated to an agent, decrementing reference counts.
        
        Returns the number of blocks purged completely from memory.
        """
        if agent_id not in self._agent_allocations:
            return 0

        purged_count = 0
        allocated_ids = self._agent_allocations.pop(agent_id)

        for content_hash, block in list(self._shared_registry.items()):
            if block.block_id in allocated_ids:
                block.reference_count -= 1
                if block.reference_count <= 0:
                    del self._shared_registry[content_hash]
                    purged_count += 1

        return purged_count

    def get_mesh_summary(self) -> CrossAgentMeshSummary:
        """Calculates total swarm deduplication metrics and memory savings."""
        active_agents = len(self._agent_allocations)
        total_agent_blocks = sum(len(b_set) for b_set in self._agent_allocations.values())
        unique_blocks = len(self._shared_registry)

        total_tokens = sum(b.token_count for b in self._shared_registry.values()) * max(1, active_agents)
        unique_tokens = sum(b.token_count for b in self._shared_registry.values())

        if total_agent_blocks > 0:
            dedupe_ratio = ((total_agent_blocks - unique_blocks) / total_agent_blocks) * 100.0
        else:
            dedupe_ratio = 0.0

        # Memory reclaimed in GB
        reclaimed_bytes = sum(
            b.raw_bytes * max(0, b.reference_count - 1)
            for b in self._shared_registry.values()
        )
        reclaimed_gb = reclaimed_bytes / (1024.0 * 1024.0 * 1024.0)

        return CrossAgentMeshSummary(
            active_subagents=active_agents,
            total_swarm_tokens=total_tokens,
            unique_tokens_retained=unique_tokens,
            deduplication_ratio_pct=round(max(0.0, dedupe_ratio), 2),
            shared_blocks_count=unique_blocks,
            hbm_reclaimed_gb=round(reclaimed_gb, 3),
        )
