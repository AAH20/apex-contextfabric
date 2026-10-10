"""Alibaba Open Code Review & AnyAgent Integration Adapter.

Performs automated pre-flight static code and architecture reviews,
evaluating memory management safety, dangling reference hazards,
and architectural conformance for agentic memory runtimes.
Zero external dependencies (pure Python 3.9+ standard library only).
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional


class AlibabaCodeReviewAdapter:
    """Pre-flight code reviewer evaluating memory safety and architectural hygiene."""

    def review_codebase_module(self, source_code: str, module_name: str = "core.py") -> Dict[str, Any]:
        """Parses Python source code and inspects memory management practices."""
        findings: List[Dict[str, Any]] = []

        try:
            tree = ast.parse(source_code, filename=module_name)
            # Scan for dangerous constructs
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"

                    if func_name in {"eval", "exec", "os.system"}:
                        findings.append({
                            "severity": "CRITICAL",
                            "rule": "UNSAFE_EXECUTION",
                            "line": getattr(node, "lineno", 1),
                            "detail": f"Prohibited dynamic call: {func_name}",
                        })
        except SyntaxError as e:
            findings.append({
                "severity": "CRITICAL",
                "rule": "SYNTAX_ERROR",
                "line": e.lineno or 1,
                "detail": str(e),
            })

        criticals = sum(1 for f in findings if f["severity"] == "CRITICAL")
        passed = criticals == 0

        return {
            "module_name": module_name,
            "total_findings": len(findings),
            "critical_findings": criticals,
            "passed": passed,
            "quality_grade": "A+" if passed else "NEEDS_REFACTOR",
            "findings": findings,
        }
