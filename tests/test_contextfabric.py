"""Comprehensive unit and integration test suite for Apex_ContextFabric.

Verifies KVTC transform coder, CMX hierarchical tiering, cross-agent mesh deduplication,
attention sink pruner, enterprise adapters, orchestrator workflows, and CLI commands.
100% deterministic, zero external dependencies (pure Python standard library).
"""

import io
import unittest
from contextlib import redirect_stdout

from apex_contextfabric.core.primitives import (
    MemoryTier,
    QuantizationPrecision,
)
from apex_contextfabric.core.kvtc_transform_coder import KVTCTransformCoder
from apex_contextfabric.core.cmx_hierarchical_tier import CMXHierarchicalTierManager
from apex_contextfabric.core.cross_agent_mesh_dedupe import CrossAgentMeshDeduplicator
from apex_contextfabric.core.attention_sink_pruner import AttentionSinkPruner

from apex_contextfabric.adapters.treecompute_adapter import TreeComputePrefixAdapter
from apex_contextfabric.adapters.control_plane_adapter import ControlPlaneFinOpsAdapter
from apex_contextfabric.adapters.integration_platform_adapter import FlowForgeContextReplayAdapter
from apex_contextfabric.adapters.clef_adapter import ClefEdgeReplicator
from apex_contextfabric.adapters.paperclip_adapter import PaperclipBudgetGovernor
from apex_contextfabric.adapters.aerial_6g_connector import Aerial6GContextConnector
from apex_contextfabric.adapters.alibaba_review_adapter import AlibabaCodeReviewAdapter

from apex_contextfabric.orchestrator import ApexContextFabricOrchestrator
from apex_contextfabric.cli import main


class TestApexContextFabric(unittest.TestCase):

    def setUp(self):
        self.orchestrator = ApexContextFabricOrchestrator(
            target_compression_ratio=32.0,
            hbm_capacity_mb=100.0,  # Compact ceiling to test eviction easily
            default_budget_usd=10.00,
        )

    # 1. KVTC Transform Coder Tests
    def test_kvtc_transform_coder_compression_ratio(self):
        coder = KVTCTransformCoder(target_compression_ratio=32.0)
        desc, metrics = coder.compress_kv_block(
            block_id="blk-test-01",
            token_count=4096,
            head_count=32,
            head_dim=128,
            is_attention_sink=False,
        )
        self.assertGreater(metrics.compression_ratio, 20.0)
        self.assertGreater(metrics.memory_savings_pct, 90.0)
        self.assertLess(metrics.perplexity_delta, 0.05)
        self.assertEqual(desc.precision, QuantizationPrecision.INT4_TRANSFORM)
        self.assertGreater(desc.raw_bytes, desc.compressed_bytes)

    def test_kvtc_attention_sink_handling(self):
        coder = KVTCTransformCoder()
        desc, metrics = coder.compress_kv_block(
            block_id="blk-sink-01",
            token_count=16,
            is_attention_sink=True,
        )
        self.assertEqual(desc.precision, QuantizationPrecision.FP8_STANDARD)
        self.assertEqual(metrics.compression_ratio, 2.0)
        self.assertTrue(desc.is_attention_sink)

    def test_kvtc_decompression_roundtrip(self):
        coder = KVTCTransformCoder()
        desc, _ = coder.compress_kv_block(block_id="blk-roundtrip", token_count=1024)
        rec = coder.decompress_kv_block(desc)
        self.assertEqual(rec["block_id"], "blk-roundtrip")
        self.assertEqual(rec["reconstructed_tokens"], 1024)

    # 2. CMX Hierarchical Tiering Tests
    def test_cmx_hierarchical_tier_hbm_allocation(self):
        tier_mgr = CMXHierarchicalTierManager(hbm_capacity_mb=1000.0)
        coder = KVTCTransformCoder()
        desc, _ = coder.compress_kv_block(block_id="blk-hbm-01", token_count=2048)
        assigned_tier = tier_mgr.allocate_block(desc)
        self.assertEqual(assigned_tier, MemoryTier.TIER0_HBM3E)
        retrieved = tier_mgr.get_block("blk-hbm-01")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.block_id, "blk-hbm-01")

    def test_cmx_hierarchical_tier_lru_eviction_and_prefetch(self):
        # Set tiny HBM capacity to force eviction
        tier_mgr = CMXHierarchicalTierManager(hbm_capacity_mb=2.0, hbm_eviction_threshold_pct=50.0)
        coder = KVTCTransformCoder()

        desc1, _ = coder.compress_kv_block(block_id="blk-1", token_count=8192)
        desc2, _ = coder.compress_kv_block(block_id="blk-2", token_count=8192)
        desc3, _ = coder.compress_kv_block(block_id="blk-3", token_count=8192)

        tier_mgr.allocate_block(desc1)
        tier_mgr.allocate_block(desc2)
        tier_mgr.allocate_block(desc3)

        status = tier_mgr.get_tiering_status()
        self.assertGreater(status.tier1_ram_mb, 0.0)

        # Prefetch evicted block from RAM
        retrieved1 = tier_mgr.get_block("blk-1")
        self.assertIsNotNone(retrieved1)
        self.assertEqual(retrieved1.tier, MemoryTier.TIER0_HBM3E)

    # 3. Cross-Agent Mesh Deduplication Tests
    def test_cross_agent_mesh_deduplication(self):
        dedupe = CrossAgentMeshDeduplicator()
        coder = KVTCTransformCoder()
        desc, _ = coder.compress_kv_block(block_id="shared-blk", token_count=16384)

        # Agent 1 registers block
        b1, is_dedup1 = dedupe.register_agent_block("agent-1", "content-hash-alpha", desc)
        self.assertFalse(is_dedup1)
        self.assertEqual(b1.reference_count, 1)

        # Agent 2 registers identical block
        b2, is_dedup2 = dedupe.register_agent_block("agent-2", "content-hash-alpha", desc)
        self.assertTrue(is_dedup2)
        self.assertEqual(b2.reference_count, 2)

        summary = dedupe.get_mesh_summary()
        self.assertEqual(summary.active_subagents, 2)
        self.assertGreater(summary.deduplication_ratio_pct, 0.0)

    def test_cross_agent_mesh_release_lifecycle(self):
        dedupe = CrossAgentMeshDeduplicator()
        coder = KVTCTransformCoder()
        desc, _ = coder.compress_kv_block(block_id="ephemeral-blk", token_count=4096)

        dedupe.register_agent_block("agent-1", "hash-beta", desc)
        dedupe.register_agent_block("agent-2", "hash-beta", desc)

        # Release agent-1
        purged1 = dedupe.release_agent_context("agent-1")
        self.assertEqual(purged1, 0)  # Still retained for agent-2

        # Release agent-2
        purged2 = dedupe.release_agent_context("agent-2")
        self.assertEqual(purged2, 1)  # Now purged

    # 4. Attention Sink Pruner Tests
    def test_attention_sink_pruner_boundaries(self):
        pruner = AttentionSinkPruner(sink_token_count=16, entropy_prune_threshold=0.80)
        self.assertTrue(pruner.is_sink_index(0))
        self.assertTrue(pruner.is_sink_index(15))
        self.assertFalse(pruner.is_sink_index(16))

        retained, pruned, ratio = pruner.filter_token_stream(token_count=32768, sliding_window_size=4096)
        self.assertGreater(pruned, 0)
        self.assertGreater(retained, 4096)
        self.assertLess(ratio, 100.0)

    # 5. Enterprise Adapters Tests
    def test_treecompute_adapter_binding(self):
        adapter = TreeComputePrefixAdapter(group_id="test-grpo-group")
        rec = adapter.bind_compressed_prefix(
            slot_id="slot-001",
            kv_block_id="blk-kv-001",
            token_count=8192,
            compression_ratio=32.4,
        )
        self.assertEqual(rec["status"], "BOUND_SHARED_PREFIX_ACTIVE")
        self.assertEqual(adapter.get_slot_count(), 1)

    def test_control_plane_finops_adapter(self):
        finops = ControlPlaneFinOpsAdapter(cluster_name="test-cluster")
        finops.log_hbm_memory_savings("node-1", uncompressed_gb=64.0, compressed_gb=2.0)
        summary = finops.get_finops_summary()
        self.assertEqual(summary["cumulative_hbm_reclaimed_gb"], 62.0)
        self.assertGreater(summary["cumulative_dollar_savings"], 0.0)

    def test_flowforge_context_replay_checkpoint(self):
        replay = FlowForgeContextReplayAdapter()
        rec = replay.create_context_checkpoint("chk-01", "sess-01", ["blk-1", "blk-2"], 16384)
        self.assertTrue(rec["rollback_ready"])
        self.assertTrue(replay.rollback_to_checkpoint("chk-01"))
        self.assertFalse(replay.rollback_to_checkpoint("chk-nonexistent"))

    def test_clef_edge_replicator(self):
        clef = ClefEdgeReplicator()
        rec = clef.broadcast_block_metadata("blk-edge", "sha256hash", 32.0, 4096)
        self.assertEqual(rec["consensus_state"], "COMMITTED_SUB_100US")
        self.assertTrue(clef.is_block_known("blk-edge"))

    def test_paperclip_budget_governor_deductions_and_exhaustion(self):
        governor = PaperclipBudgetGovernor(default_budget_usd=0.20)
        receipt = governor.charge_memory_operation("TEST_OP", 50_000.0, {"token": 1000})  # $0.05
        self.assertEqual(receipt["heartbeat_state"], "ACTIVE_HEALTHY")
        self.assertAlmostEqual(governor.get_remaining_budget(), 0.15, places=4)

        with self.assertRaises(RuntimeError):
            governor.charge_memory_operation("EXCESS_OP", 300_000.0, {})

    def test_aerial_6g_context_connector(self):
        aerial = Aerial6GContextConnector(max_sla_ms=5.0)
        aerial.log_stream_event("blk-stream-1", 1.85, 2048)
        aerial.log_stream_event("blk-stream-2", 2.45, 4096)
        status = aerial.get_urllc_status()
        self.assertEqual(status["status"], "URLLC_COMPLIANT")
        self.assertEqual(status["compliance_rate_pct"], 100.0)

    def test_alibaba_code_review_adapter(self):
        reviewer = AlibabaCodeReviewAdapter()
        res_clean = reviewer.review_codebase_module("def compute():\n    return 42\n")
        self.assertTrue(res_clean["passed"])
        self.assertEqual(res_clean["quality_grade"], "A+")

        res_dirty = reviewer.review_codebase_module("exec('import os')\n")
        self.assertFalse(res_dirty["passed"])

    # 6. Orchestrator End-to-End Tests
    def test_orchestrator_compress_and_store(self):
        state = self.orchestrator.compress_and_store_context(
            session_id="test-session-e2e",
            token_count=65536,
            agent_id="agent-01",
        )
        self.assertEqual(state.context_tokens, 65536)
        self.assertGreater(state.kvtc_metrics.compression_ratio, 20.0)
        self.assertGreater(state.cmx_status.tier0_hbm_mb, 0.0)
        self.assertIsNotNone(state.ledger_sha256)

    def test_orchestrator_swarm_simulation(self):
        summary = self.orchestrator.simulate_swarm_deduplication(
            subagent_count=8,
            shared_prompt_tokens=32768,
            private_tokens_per_agent=4096,
        )
        self.assertEqual(summary.active_subagents, 8)
        self.assertGreater(summary.deduplication_ratio_pct, 40.0)
        self.assertGreater(summary.hbm_reclaimed_gb, 0.0)

    def test_orchestrator_markdown_report(self):
        state = self.orchestrator.compress_and_store_context(
            session_id="test-report-session",
            token_count=32768,
        )
        report = self.orchestrator.generate_executive_markdown_report(state)
        self.assertIn("# Apex_ContextFabric", report)
        self.assertIn("KV-Cache Transform Coding", report)
        self.assertIn("Hierarchical Context Storage", report)

    # 7. CLI Subcommand Tests
    def test_cli_subcommands(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret_comp = main(["compress", "--tokens", "4096"])
            self.assertEqual(ret_comp, 0)

            ret_tier = main(["tiering"])
            self.assertEqual(ret_tier, 0)

            ret_swarm = main(["swarm-dedupe", "--agents", "4"])
            self.assertEqual(ret_swarm, 0)

            ret_finops = main(["finops"])
            self.assertEqual(ret_finops, 0)

            ret_report = main(["report"])
            self.assertEqual(ret_report, 0)


if __name__ == "__main__":
    unittest.main()
