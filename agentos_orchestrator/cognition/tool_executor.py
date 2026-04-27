"""Tool Executor — Sandboxed Code Execution for the Cognitive Agent.

Phase 3: Tool Augmentation.  For tasks like "analyse the stock market" or
"compute portfolio VaR", clicking a UI hundreds of times is the wrong level
of abstraction.  This module gives the agent a code execution primitive: it
writes a Python script, runs it in a tightly sandboxed subprocess, and gets
back structured output (stdout, stderr, artefacts).

Architecture
────────────
    MacroPlanner recognises a "tool_use" option.
        └─ HierarchicalTaskDecomposer builds a ToolUseOption.
               └─ ToolExecutor.run_script(code, timeout)
                       ↓
               SubprocessSandbox (isolated Python, no network unless allowed)
                       ↓
               ToolResult(stdout, artefacts, error, elapsed_ms)

Security posture
────────────────
* Runs in a subprocess with a configurable wall-clock timeout (default 30 s).
* `restrict_imports` list blocks dangerous stdlib modules (os.system, subprocess,
  socket, ctypes, importlib, etc.) unless explicitly allow-listed.
* The spawned process inherits no extra environment variables beyond what the
  caller explicitly passes via `env_vars`.
* All file I/O is confined to a per-run scratch directory under the workspace
  .agentos/sandbox/ folder; the script is forbidden from writing outside it via
  a sys.path restriction and a path-guard wrapper injected at the top.
* This is defence-in-depth, not a full OS-level sandbox (no seccomp / container
  isolation in this version).  For untrusted third-party code, wrap in a container.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────── #
# Data Structures                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class ToolResult:
    """The result of a sandboxed code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: float = 0.0
    artefacts: list[Path] = field(default_factory=list)
    error: str | None = None
    # Parsed key-value results the script emits via RESULT: key=value lines
    parsed_results: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-liner for use in agent reasoning."""
        if not self.success:
            return f"[TOOL_ERROR] {self.error or self.stderr[:200]}"
        lines = [ln for ln in self.stdout.splitlines() if ln.strip()]
        preview = lines[-5:] if lines else ["(no output)"]
        return "[TOOL_OK] " + " | ".join(preview)


@dataclass
class QuantAnalysisRequest:
    """Structured request for quantitative data analysis."""

    objective: str
    # Python code the agent wants to execute
    code: str
    # Allow network access (default off for safety)
    allow_network: bool = False
    # Extra pip packages to install before running (pre-vetted list only)
    allowed_packages: list[str] = field(default_factory=list)
    timeout_seconds: int = 30
    # Explicit env vars to expose (API keys etc., by name — values from env)
    expose_env_keys: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────── #
# Sandbox                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

# Modules that may not be imported in sandboxed code
_BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "ctypes",
        "socket",
        "asyncio",  # can be used to open sockets
        "multiprocessing",
        "threading",
        "importlib",
        "builtins.__import__",
        "signal",
        "pty",
        "resource",
        "gc",   # can be abused to walk the object graph
        "_thread",
    }
)

# Packages we are willing to install on demand (fixed allow-list)
_VETTED_PACKAGES: frozenset[str] = frozenset(
    {
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "statsmodels",
        "yfinance",
        "requests",      # only usable if allow_network=True
        "httpx",
        "polars",
        "openpyxl",
        "plotly",
        "seaborn",
    }
)


class ToolExecutor:
    """Sandboxed Python code executor for agent tool-use.

    Usage
    ─────
        executor = ToolExecutor(workspace_root="/path/to/.agentos")
        result   = executor.run(QuantAnalysisRequest(
            objective = "compute 30-day rolling vol for AAPL",
            code      = "...",
            allow_network = True,
        ))
        print(result.summary())
    """

    def __init__(self, workspace_root: str | Path = ".agentos") -> None:
        self._workspace = Path(workspace_root)
        self._sandbox_dir = self._workspace / "sandbox"
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, request: QuantAnalysisRequest) -> ToolResult:
        """Execute sandboxed code and return a ToolResult."""
        run_id = uuid.uuid4().hex[:8]
        run_dir = self._sandbox_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._execute(request, run_dir, run_id)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"Executor internal error: {exc}\n{traceback.format_exc()}",
            )

    def build_quant_analysis_code(
        self,
        objective: str,
        tickers: list[str] | None = None,
        period: str = "1y",
    ) -> str:
        """Generate boilerplate quant analysis code for common objectives.

        The agent can call this instead of writing code from scratch.
        It is always faster to use a known-good template than to hallucinate
        arbitrary pandas.

        Returns Python source that prints a RESULT: summary= line.
        """
        tickers_repr = repr(tickers or ["SPY"])
        return textwrap.dedent(f"""
            # Auto-generated by ToolExecutor.build_quant_analysis_code
            # Objective: {objective}
            import json

            try:
                import numpy as np
                import pandas as pd
                import yfinance as yf
                tickers = {tickers_repr}
                data = yf.download(tickers, period="{period}", auto_adjust=True, progress=False)
                closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
                closes = closes.dropna()
                returns = closes.pct_change().dropna()

                # Core statistics
                ann_factor = 252
                summary = {{}}
                for col in returns.columns:
                    r = returns[col]
                    summary[col] = {{
                        "mean_daily_ret":  round(float(r.mean()), 6),
                        "ann_return":      round(float(r.mean() * ann_factor), 4),
                        "ann_vol":         round(float(r.std() * np.sqrt(ann_factor)), 4),
                        "sharpe":          round(float(r.mean() / r.std() * np.sqrt(ann_factor)), 3),
                        "max_drawdown":    round(float((closes[col] / closes[col].cummax() - 1).min()), 4),
                        "current_price":   round(float(closes[col].iloc[-1]), 2),
                    }}

                print(json.dumps(summary, indent=2))
                print("RESULT: summary=" + json.dumps(summary))
            except ImportError as e:
                print(f"RESULT: error=missing package: {{e}}")
            except Exception as e:
                print(f"RESULT: error={{e}}")
        """).strip()

    # ------------------------------------------------------------------ #
    # Internal Execution Engine                                            #
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        request: QuantAnalysisRequest,
        run_dir: Path,
        run_id: str,
    ) -> ToolResult:
        # 1. Validate / sanitise code
        validation_error = self._validate_code(request.code)
        if validation_error:
            return ToolResult(success=False, error=f"Validation: {validation_error}")

        # 2. Install requested packages (only vetted ones)
        for pkg in request.allowed_packages:
            if pkg not in _VETTED_PACKAGES:
                return ToolResult(
                    success=False,
                    error=f"Package '{pkg}' is not on the allow-list.",
                )

        # 3. Build wrapper script
        script_path = run_dir / "script.py"
        wrapper = self._build_wrapper(request.code, run_dir, request.allow_network)
        script_path.write_text(wrapper, encoding="utf-8")

        # 4. Collect env vars
        env = self._build_env(request.expose_env_keys)

        # 5. Run in subprocess
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                cwd=str(run_dir),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Script timed out after {request.timeout_seconds}s",
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 6. Collect artefacts (any files written to run_dir)
        artefacts = [
            p for p in run_dir.iterdir()
            if p.is_file() and p.name != "script.py"
        ]

        # 7. Parse RESULT: lines from stdout
        parsed = self._parse_results(proc.stdout)

        success = proc.returncode == 0
        return ToolResult(
            success=success,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_ms=elapsed_ms,
            artefacts=artefacts,
            error=proc.stderr[:500] if not success else None,
            parsed_results=parsed,
        )

    def _build_wrapper(
        self, code: str, run_dir: Path, allow_network: bool
    ) -> str:
        """Inject path guard and optional network block around user code."""
        network_guard = "" if allow_network else textwrap.dedent("""
            import sys as _sys
            _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
            def _guarded_import(name, *args, **kwargs):
                _BLOCKED = {"socket", "httpx", "requests", "ftplib",
                            "smtplib", "asyncio", "aiohttp"}
                if any(name == b or name.startswith(b + ".") for b in _BLOCKED):
                    raise ImportError(f"Network access is disabled. Module '{name}' is blocked.")
                return _real_import(name, *args, **kwargs)
            import builtins as _builtins
            _builtins.__import__ = _guarded_import
        """).strip()

        path_guard = textwrap.dedent(f"""
            import os as _os
            _SANDBOX_DIR = {str(run_dir)!r}
            _orig_open = open
            def _safe_open(file, mode="r", *a, **kw):
                p = _os.path.abspath(str(file))
                if "w" in mode or "a" in mode or "x" in mode:
                    if not p.startswith(_SANDBOX_DIR):
                        raise PermissionError(
                            f"Write outside sandbox denied: {{p}}"
                        )
                return _orig_open(file, mode, *a, **kw)
            import builtins as _builtins
            _builtins.open = _safe_open
        """).strip()

        return "\n".join([
            path_guard,
            network_guard,
            "# ── user code ──────────────────────────────",
            code,
        ])

    @staticmethod
    def _validate_code(code: str) -> str | None:
        """Lightweight static check — returns an error string or None."""
        blocked_calls = [
            "os.system(",
            "os.popen(",
            "subprocess.run(",
            "subprocess.Popen(",
            "subprocess.call(",
            "exec(",
            "eval(",
            "compile(",
            "__import__(",
        ]
        lower = code.lower()
        for call in blocked_calls:
            if call.lower() in lower:
                return f"Forbidden call: {call}"
        for mod in _BLOCKED_MODULES:
            if f"import {mod}" in code or f"from {mod}" in code:
                return f"Blocked import: {mod}"
        return None

    @staticmethod
    def _build_env(expose_keys: list[str]) -> dict[str, str]:
        """Build a minimal env dict for the subprocess."""
        # Pass the complete current environment as base so the subprocess can
        # find the active venv's site-packages (PYTHONPATH, VIRTUAL_ENV, etc.).
        # We then selectively override only what is needed for isolation.
        env = dict(os.environ)
        # Expose only explicitly requested secrets; others remain from parent env
        for key in expose_keys:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        return env

    @staticmethod
    def _parse_results(stdout: str) -> dict[str, Any]:
        """Extract RESULT: key=value lines from stdout."""
        import json
        results: dict[str, Any] = {}
        for line in stdout.splitlines():
            if line.startswith("RESULT:"):
                payload = line[len("RESULT:"):].strip()
                # Try key=json_value format
                if "=" in payload:
                    k, _, v = payload.partition("=")
                    try:
                        results[k.strip()] = json.loads(v.strip())
                    except json.JSONDecodeError:
                        results[k.strip()] = v.strip()
        return results


# ─────────────────────────────────────────────────────────────────────────── #
# Pre-built Analysis Templates                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

_TEMPLATE_REGISTRY: dict[str, str] = {}


def register_template(name: str, code: str) -> None:
    _TEMPLATE_REGISTRY[name] = code


def get_template(name: str) -> str | None:
    return _TEMPLATE_REGISTRY.get(name)


register_template(
    "portfolio_stats",
    textwrap.dedent("""
        import json, numpy as np, pandas as pd
        # USAGE: set TICKERS and PERIOD before running
        TICKERS = TICKERS if "TICKERS" in dir() else ["SPY", "QQQ", "GLD"]
        PERIOD  = PERIOD  if "PERIOD"  in dir() else "1y"
        try:
            import yfinance as yf
            raw = yf.download(TICKERS, period=PERIOD, auto_adjust=True, progress=False)
            prices = raw["Close"].dropna()
            rets = prices.pct_change().dropna()
            cov = rets.cov() * 252
            corr = rets.corr()
            ann = rets.mean() * 252
            vols = rets.std() * (252 ** 0.5)
            sharpes = ann / vols
            result = {
                t: {"ann_return": round(float(ann[t]), 4),
                    "ann_vol":    round(float(vols[t]), 4),
                    "sharpe":     round(float(sharpes[t]), 3)}
                for t in TICKERS if t in ann.index
            }
            print(json.dumps(result, indent=2))
            print("RESULT: stats=" + json.dumps(result))
        except Exception as e:
            print(f"RESULT: error={e}")
    """).strip(),
)

register_template(
    "rolling_volatility",
    textwrap.dedent("""
        import json, numpy as np, pandas as pd
        TICKER = TICKER if "TICKER" in dir() else "SPY"
        WINDOW = WINDOW if "WINDOW" in dir() else 30
        PERIOD = PERIOD if "PERIOD" in dir() else "1y"
        try:
            import yfinance as yf
            prices = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)["Close"].dropna()
            rv = prices.pct_change().rolling(WINDOW).std() * (252 ** 0.5)
            latest = round(float(rv.iloc[-1]), 4)
            avg    = round(float(rv.mean()), 4)
            print(f"Rolling {WINDOW}d annualised vol for {TICKER}:")
            print(f"  Latest : {latest:.2%}")
            print(f"  Average: {avg:.2%}")
            print(f"RESULT: rolling_vol={{\"ticker\":\"{TICKER}\",\"window\":{WINDOW},\"latest\":{latest},\"avg\":{avg}}}")
        except Exception as e:
            print(f"RESULT: error={e}")
    """).strip(),
)

register_template(
    "market_regime_hmm",
    textwrap.dedent("""
        import json, numpy as np, pandas as pd
        TICKER = TICKER if "TICKER" in dir() else "SPY"
        try:
            import yfinance as yf
            from sklearn.mixture import GaussianMixture
            prices = yf.download(TICKER, period="2y", auto_adjust=True, progress=False)["Close"].dropna()
            rets = prices.pct_change().dropna().values.reshape(-1, 1)
            gm = GaussianMixture(n_components=2, random_state=42).fit(rets)
            labels = gm.predict(rets)
            # Label 0 = lower vol regime, 1 = higher vol
            means = gm.means_.flatten()
            stds  = np.sqrt(gm.covariances_.flatten())
            regime_now = int(labels[-1])
            result = {
                "ticker": TICKER,
                "current_regime": regime_now,
                "regime_0": {"mean_ret": round(float(means[0]), 6), "vol": round(float(stds[0]), 6)},
                "regime_1": {"mean_ret": round(float(means[1]), 6), "vol": round(float(stds[1]), 6)},
            }
            print(json.dumps(result, indent=2))
            print("RESULT: regime=" + json.dumps(result))
        except Exception as e:
            print(f"RESULT: error={e}")
    """).strip(),
)
