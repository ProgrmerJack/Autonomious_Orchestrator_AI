from __future__ import annotations

import base64
import json
import platform
import shutil
import subprocess
from pathlib import Path

from .base import BackendUnavailable, UiAction, UiNode


class WindowsUiaBackend:
    """Windows UI Automation backend through PowerShell/.NET."""

    name = "windows-uia"

    def __init__(
        self,
        powershell_path: str | None = None,
        timeout_seconds: int = 15,
        max_depth: int = 3,
        max_nodes: int = 500,
    ) -> None:
        self.powershell_path = powershell_path or self._find_powershell()
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def available(self) -> bool:
        is_windows = platform.system() == "Windows"
        return is_windows and self.powershell_path is not None

    def snapshot(self) -> list[UiNode]:
        self._ensure_available()
        payload = self._run_json(self._snapshot_script())
        nodes: list[UiNode] = []
        for item in payload.get("nodes", []):
            bounds = None
            if item.get("x") is not None:
                bounds = (
                    int(item["x"]),
                    int(item["y"]),
                    int(item["width"]),
                    int(item["height"]),
                )
            nodes.append(
                UiNode(
                    node_id=str(item.get("node_id", "")),
                    role=str(item.get("role", "unknown")).replace(
                        "ControlType.",
                        "",
                    ),
                    name=str(item.get("name", "")),
                    bounds=bounds,
                    enabled=bool(item.get("enabled", True)),
                    focused=bool(item.get("focused", False)),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        return nodes

    def perform(self, action: UiAction) -> str:
        self._ensure_available()
        payload = self._run_json(
            self._perform_script(
                action.action_type,
                action.selector,
                action.value,
            )
        )
        return json.dumps(payload, sort_keys=True)

    def _run_json(self, script: str) -> dict:
        if self.powershell_path is None:
            raise BackendUnavailable("PowerShell is not available")
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        result = subprocess.run(
            [
                self.powershell_path,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise BackendUnavailable(result.stderr.strip() or result.stdout)
        output = result.stdout.strip()
        if not output:
            return {}
        return json.loads(output)

    def _snapshot_script(self) -> str:
        return f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
$MaxDepth = {self.max_depth}
$MaxNodes = {self.max_nodes}
$Items = New-Object System.Collections.Generic.List[object]

function Add-Node($Element, [int]$Depth, [string]$ParentId) {{
  if ($Items.Count -ge $MaxNodes) {{ return }}
  try {{
    $Current = $Element.Current
    $Rect = $Current.BoundingRectangle
    $NodeId = ('{{0}}:{{1}}:{{2}}' -f `
      $Depth,
      $Items.Count,
      $Current.AutomationId
    )
    $Items.Add([pscustomobject]@{{
      node_id = $NodeId
      role = $Current.ControlType.ProgrammaticName
      name = $Current.Name
      x = [int]$Rect.X
      y = [int]$Rect.Y
      width = [int]$Rect.Width
      height = [int]$Rect.Height
      enabled = [bool]$Current.IsEnabled
      focused = [bool]$Current.HasKeyboardFocus
      metadata = @{{
        automation_id = $Current.AutomationId
        class_name = $Current.ClassName
        process_id = $Current.ProcessId
        parent = $ParentId
      }}
    }})
    if ($Depth -ge $MaxDepth) {{ return }}
    $Children = $Element.FindAll(
      [System.Windows.Automation.TreeScope]::Children,
      [System.Windows.Automation.Condition]::TrueCondition
    )
    foreach ($Child in $Children) {{ Add-Node $Child ($Depth + 1) $NodeId }}
  }} catch {{ }}
}}

$Root = [System.Windows.Automation.AutomationElement]::RootElement
$Windows = $Root.FindAll(
  [System.Windows.Automation.TreeScope]::Children,
  [System.Windows.Automation.Condition]::TrueCondition
)
foreach ($Window in $Windows) {{ Add-Node $Window 0 '' }}
[pscustomobject]@{{ nodes = $Items }} | ConvertTo-Json -Depth 8 -Compress
"""

    def _perform_script(
        self,
        action_type: str,
        selector: str,
        value: str | None,
    ) -> str:
        selector_b64 = self._b64(selector)
        action_b64 = self._b64(action_type)
        value_b64 = self._b64(value or "")
        return f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName System.Windows.Forms
$Selector = [Text.Encoding]::UTF8.GetString(
  [Convert]::FromBase64String('{selector_b64}')
)
$ActionType = [Text.Encoding]::UTF8.GetString(
  [Convert]::FromBase64String('{action_b64}')
)
$Value = [Text.Encoding]::UTF8.GetString(
  [Convert]::FromBase64String('{value_b64}')
)
$IgnoreCase = [System.StringComparison]::OrdinalIgnoreCase

function Get-Needle($Prefix) {{
  if ($Selector.StartsWith($Prefix)) {{
    return $Selector.Substring($Prefix.Length)
  }}
  return $null
}}

function Matches($Element) {{
  try {{
    $Current = $Element.Current
    $Name = [string]$Current.Name
    $AutomationId = [string]$Current.AutomationId
    $Role = [string]$Current.ControlType.ProgrammaticName
    $ClassName = [string]$Current.ClassName
    $Needle = Get-Needle 'name='
    if ($Needle -and $Name.IndexOf($Needle, $IgnoreCase) -ge 0) {{
      return $true
    }}
    $Needle = Get-Needle 'automation_id='
    if ($Needle -and
      $AutomationId.IndexOf($Needle, $IgnoreCase) -ge 0) {{
      return $true
    }}
    $Needle = Get-Needle 'role='
    if ($Needle -and $Role.IndexOf($Needle, $IgnoreCase) -ge 0) {{
      return $true
    }}
    if (-not $Selector.Contains('=')) {{
      return (
        $Name.IndexOf($Selector, $IgnoreCase) -ge 0 -or
        $AutomationId.IndexOf($Selector, $IgnoreCase) -ge 0 -or
        $ClassName.IndexOf($Selector, $IgnoreCase) -ge 0
      )
    }}
  }} catch {{ }}
  return $false
}}

function Find-Target() {{
  $Root = [System.Windows.Automation.AutomationElement]::RootElement
  $Queue = New-Object System.Collections.Queue
  $Children = $Root.FindAll(
    [System.Windows.Automation.TreeScope]::Children,
    [System.Windows.Automation.Condition]::TrueCondition
  )
  foreach ($Child in $Children) {{ $Queue.Enqueue($Child) }}
  $Visited = 0
  while ($Queue.Count -gt 0 -and $Visited -lt {self.max_nodes * 10}) {{
    $Visited += 1
    $Element = $Queue.Dequeue()
    if (Matches $Element) {{ return $Element }}
    try {{
      $Children = $Element.FindAll(
        [System.Windows.Automation.TreeScope]::Children,
        [System.Windows.Automation.Condition]::TrueCondition
      )
      foreach ($Child in $Children) {{ $Queue.Enqueue($Child) }}
    }} catch {{ }}
  }}
  return $null
}}

$Target = Find-Target
if ($null -eq $Target) {{ throw "No UI element matched selector '$Selector'" }}
$FocusError = $null
$Status = 'matched'
try {{
  $Target.SetFocus()
  $Status = 'focused'
}} catch {{
  $FocusError = $_.Exception.Message
}}
if ($ActionType -eq 'invoke' -or $ActionType -eq 'click') {{
  $Pattern = $null
  if ($Target.TryGetCurrentPattern(
      [System.Windows.Automation.InvokePattern]::Pattern,
      [ref]$Pattern
  )) {{
    $Pattern.Invoke()
    $Status = 'invoked'
  }}
}}
if ($ActionType -eq 'set_text' -or $ActionType -eq 'type') {{
  $Pattern = $null
  if ($Target.TryGetCurrentPattern(
      [System.Windows.Automation.ValuePattern]::Pattern,
      [ref]$Pattern
  )) {{
    $Pattern.SetValue($Value)
    $Status = 'value-set'
  }} else {{
    [System.Windows.Forms.SendKeys]::SendWait($Value)
    $Status = 'typed'
  }}
}}
[pscustomobject]@{{
  status = $Status
  action_type = $ActionType
  selector = $Selector
  matched_name = $Target.Current.Name
  matched_role = $Target.Current.ControlType.ProgrammaticName
  focus_error = $FocusError
}} | ConvertTo-Json -Depth 4 -Compress
"""

    def _ensure_available(self) -> None:
        if not self.available():
            raise BackendUnavailable("Windows UI Automation is not available")

    @staticmethod
    def _b64(value: str) -> str:
        return base64.b64encode(value.encode("utf-8")).decode("ascii")

    @staticmethod
    def _find_powershell() -> str | None:
        for name in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
            path = shutil.which(name)
            if path:
                return str(Path(path))
        return None
