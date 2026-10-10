"""Command-Line Interface for Apex_ContextFabric.

Institutional CLI for KV-Cache Transform Coding (KVTC), CMX Hierarchical Tiering,
and Multi-Agent Swarm Attention Deduplication.
Zero external dependencies (pure Python 3.9+ standard library only).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from apex_contextfabric.orchestrator import ApexContextFabricOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextfabric",
        description="Apex_ContextFabric: Sovereign 32x KV-Cache Transform Coding & Context Memory Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command 1: compress
    comp_parser = subparsers.add_parser("compress", help="Compress KV cache block via 32x transform coding")
    comp_parser.add_argument("--tokens", type=int, default=131072, help="Context length in tokens")
    comp_parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    comp_parser.add_argument("--dim", type=int, default=128, help="Dimension per head")
    comp_parser.add_argument("--model", type=str, default="nvidia/nemotron-3-ultra-550b", help="Target model name")

    # Command 2: tiering
    subparsers.add_parser("tiering", help="Display CMX 3-tier memory allocation (HBM, Host RAM, NVMe)")

    # Command 3: swarm-dedupe
    dedupe_parser = subparsers.add_parser("swarm-dedupe", help="Simulate cross-agent swarm attention deduplication")
    dedupe_parser.add_argument("--agents", type=int, default=16, help="Number of concurrent subagents")
    dedupe_parser.add_argument("--shared-tokens", type=int, default=65536, help="Shared base prompt tokens")
    dedupe_parser.add_argument("--private-tokens", type=int, default=8192, help="Private tokens per agent")

    # Command 4: finops
    subparsers.add_parser("finops", help="Display GPU FinOps HBM dollar savings and cluster telemetry")

    # Command 5: report
    subparsers.add_parser("report", help="Generate full executive markdown audit memo")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    orchestrator = ApexContextFabricOrchestrator()

    if args.command == "compress":
        state = orchestrator.compress_and_store_context(
            session_id="cli-compress-session",
            token_count=args.tokens,
            head_count=args.heads,
            head_dim=args.dim,
            model_name=args.model,
        )
        print("KVTC Transform Coding Completed:")
        print(f"  Target Model        : {state.model_name}")
        print(f"  Context Tokens      : {state.context_tokens:,} tokens")
        print(f"  Original FP16 Memory: {state.kvtc_metrics.original_memory_mb:,.2f} MB")
        print(f"  Compressed Memory   : {state.kvtc_metrics.compressed_memory_mb:,.2f} MB")
        print(f"  Compression Ratio   : {state.kvtc_metrics.compression_ratio}x")
        print(f"  HBM Memory Savings  : {state.kvtc_metrics.memory_savings_pct}%")
        print(f"  Perplexity Delta    : {state.kvtc_metrics.perplexity_delta}%")
        print(f"  Encoding Latency    : {state.kvtc_metrics.transform_latency_us} µs")
        print(f"  ActionLedger SHA    : {state.ledger_sha256}")
        return 0

    elif args.command == "tiering":
        # Simulate loading several blocks
        orchestrator.compress_and_store_context(session_id="session-tier-1", token_count=65536)
        orchestrator.compress_and_store_context(session_id="session-tier-2", token_count=65536)
        status = orchestrator.cmx_tier.get_tiering_status()
        print("Context Memory Storage (CMX) Tiering Status:")
        print(f"  Tier 0 (HBM3e)      : {status.tier0_hbm_mb:,.2f} MB / {status.hbm_capacity_limit_mb:,.0f} MB")
        print(f"  Tier 1 (Host RAM)   : {status.tier1_ram_mb:,.2f} MB")
        print(f"  Tier 2 (NVMe Flash) : {status.tier2_nvme_mb:,.2f} MB")
        print(f"  HBM Cache Hit Rate  : {status.hbm_hit_rate_pct}%")
        print(f"  Total Blocks Managed: {status.total_blocks_managed}")
        print(f"  Eviction / Prefetch : {status.eviction_latency_us} µs / {status.prefetch_latency_us} µs")
        return 0

    elif args.command == "swarm-dedupe":
        print(f"Simulating Cross-Agent Swarm Deduplication ({args.agents} subagents)...")
        summary = orchestrator.simulate_swarm_deduplication(
            subagent_count=args.agents,
            shared_prompt_tokens=args.shared_tokens,
            private_tokens_per_agent=args.private_tokens,
        )
        print("Swarm Deduplication Summary:")
        print(f"  Active Subagents    : {summary.active_subagents}")
        print(f"  Total Swarm Tokens  : {summary.total_swarm_tokens:,}")
        print(f"  Unique Tokens Stored: {summary.unique_tokens_retained:,}")
        print(f"  Deduplication Ratio : {summary.deduplication_ratio_pct}%")
        print(f"  HBM Reclaimed       : {summary.hbm_reclaimed_gb:.3f} GB")
        return 0

    elif args.command == "finops":
        orchestrator.control_plane.log_hbm_memory_savings("node-01", uncompressed_gb=128.0, compressed_gb=4.0)
        orchestrator.control_plane.log_hbm_memory_savings("node-02", uncompressed_gb=256.0, compressed_gb=8.0)
        finops = orchestrator.control_plane.get_finops_summary()
        print("Kubernetes GPU FinOps HBM Savings:")
        print(f"  Cluster Name        : {finops['cluster_name']}")
        print(f"  Events Logged       : {finops['events_logged']}")
        print(f"  Cumulative Reclaimed: {finops['cumulative_hbm_reclaimed_gb']:.3f} GB")
        print(f"  Hourly Dollar Yield : ${finops['cumulative_dollar_savings']:,.4f}/hr")
        return 0

    elif args.command == "report":
        state = orchestrator.compress_and_store_context(session_id="executive-demo", token_count=131072)
        orchestrator.simulate_swarm_deduplication(subagent_count=16)
        report_md = orchestrator.generate_executive_markdown_report(state)
        print(report_md)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
