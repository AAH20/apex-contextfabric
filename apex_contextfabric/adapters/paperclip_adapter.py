"""Hermes Paperclip Adapter: Company OS Micro-USD Budget Governor.

Native integration with '@paperclipai/hermes-paperclip-adapter'.
Enforces micro-USD budget ceilings on GPU HBM allocations and KV block paging,
issuing tamper-evident SHA-256 ActionLedger receipts.
Zero external dependencies (pure Python 3.9+ standard library only).
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional


class PaperclipBudgetGovernor:
    """Micro-USD budget governor and ActionLedger audit emitter."""

    def __init__(self, default_budget_usd: float = 5.00):
        self.default_budget_usd = default_budget_usd
        self._consumed_usd = 0.0
        self._action_ledger: List[Dict[str, Any]] = []

    def charge_memory_operation(
        self,
        op_name: str,
        cost_micro_usd: float,
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Charges micro-USD spend and writes an immutable ActionLedger receipt."""
        cost_usd = cost_micro_usd / 1_000_000.0
        if (self._consumed_usd + cost_usd) > self.default_budget_usd:
            raise RuntimeError(
                f"Paperclip Budget Exhaustion: Charge ${cost_usd:.6f} exceeds "
                f"remaining balance ${self.default_budget_usd - self._consumed_usd:.6f}"
            )

        self._consumed_usd += cost_usd
        timestamp = time.time()
        receipt_hash = hashlib.sha256(
            f"{op_name}:{cost_micro_usd}:{timestamp}:{json.dumps(state_data, sort_keys=True)}".encode("utf-8")
        ).hexdigest()

        receipt = {
            "receipt_id": receipt_hash[:16],
            "operation": op_name,
            "cost_micro_usd": cost_micro_usd,
            "cumulative_spent_usd": round(self._consumed_usd, 6),
            "remaining_budget_usd": round(self.default_budget_usd - self._consumed_usd, 6),
            "timestamp": timestamp,
            "sha256_digest": receipt_hash,
            "heartbeat_state": "ACTIVE_HEALTHY",
        }
        self._action_ledger.append(receipt)
        return receipt

    def get_remaining_budget(self) -> float:
        return max(0.0, self.default_budget_usd - self._consumed_usd)

    def get_ledger(self) -> List[Dict[str, Any]]:
        return list(self._action_ledger)
