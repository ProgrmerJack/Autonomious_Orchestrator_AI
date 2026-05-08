from __future__ import annotations

import base64
import binascii
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
        max_depth: int = 7,
        max_nodes: int = 2000,
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

    def capture(self) -> bytes:
        self._ensure_available()
        payload = self._run_text(self._capture_script())
        if not payload:
            return b""
        try:
            return base64.b64decode(payload)
        except (ValueError, binascii.Error) as exc:
            raise BackendUnavailable("Windows screenshot payload was invalid") from exc

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
        output = self._run_text(script)
        if not output:
            return {}
        return json.loads(output)

    def _run_text(self, script: str) -> str:
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
        return result.stdout.strip()

    def _capture_script(self) -> str:
        return """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$Bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
$Bitmap = New-Object System.Drawing.Bitmap $Bounds.Width, $Bounds.Height
$Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
try {
  $Graphics.CopyFromScreen(
    $Bounds.Left,
    $Bounds.Top,
    0,
    0,
    $Bitmap.Size,
    [System.Drawing.CopyPixelOperation]::SourceCopy
  )
  $Stream = New-Object System.IO.MemoryStream
  try {
    $Bitmap.Save($Stream, [System.Drawing.Imaging.ImageFormat]::Png)
    [Convert]::ToBase64String($Stream.ToArray())
  } finally {
    $Stream.Dispose()
  }
} finally {
  $Graphics.Dispose()
  $Bitmap.Dispose()
}
"""

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
Add-Type @"
using System.Runtime.InteropServices;
public static class AgentOSMouse {{
  [DllImport("user32.dll")]
  public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")]
  public static extern void mouse_event(
    int dwFlags,
    int dx,
    int dy,
    int data,
    int extraInfo
  );
}}
"@
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

if ($ActionType -eq 'launch_app') {{
  $LaunchTarget = if ([string]::IsNullOrWhiteSpace($Value)) {{
    $Selector
  }} else {{
    $Value
  }}
  $Process = Start-Process -FilePath $LaunchTarget -PassThru
  [pscustomobject]@{{
    status = 'launched'
    action_type = $ActionType
    selector = $Selector
    launched = $LaunchTarget
    process_id = $Process.Id
  }} | ConvertTo-Json -Depth 4 -Compress
  return
}}

if ($ActionType -eq 'hotkey') {{
  [System.Windows.Forms.SendKeys]::SendWait($Value)
  [pscustomobject]@{{
    status = 'hotkey-sent'
    action_type = $ActionType
    selector = $Selector
    value = $Value
  }} | ConvertTo-Json -Depth 4 -Compress
  return
}}

function Get-Needle($Prefix) {{
  if ($Selector.StartsWith($Prefix)) {{
    return $Selector.Substring($Prefix.Length)
  }}
  return $null
}}

function Matches-Clause($Element, [string]$Clause) {{
  try {{
    $Current = $Element.Current
    $Name = [string]$Current.Name
    $AutomationId = [string]$Current.AutomationId
    $Role = [string]$Current.ControlType.ProgrammaticName
    $ClassName = [string]$Current.ClassName
    $ProcessId = [string]$Current.ProcessId
    if ($Clause.StartsWith('name=')) {{
      $Needle = $Clause.Substring(5)
      return $Name.IndexOf($Needle, $IgnoreCase) -ge 0
    }}
    if ($Clause.StartsWith('automation_id=')) {{
      $Needle = $Clause.Substring(14)
      return $AutomationId.IndexOf($Needle, $IgnoreCase) -ge 0
    }}
    if ($Clause.StartsWith('role=')) {{
      $Needle = $Clause.Substring(5)
      return $Role.IndexOf($Needle, $IgnoreCase) -ge 0
    }}
    if ($Clause.StartsWith('class_name=')) {{
      $Needle = $Clause.Substring(11)
      return $ClassName.IndexOf($Needle, $IgnoreCase) -ge 0
    }}
    if ($Clause.StartsWith('process_id=')) {{
      $Needle = $Clause.Substring(11)
      return $ProcessId -eq $Needle
    }}
    return (
      $Name.IndexOf($Clause, $IgnoreCase) -ge 0 -or
      $AutomationId.IndexOf($Clause, $IgnoreCase) -ge 0 -or
      $ClassName.IndexOf($Clause, $IgnoreCase) -ge 0
    )
  }} catch {{ }}
  return $false
}}

function Matches($Element) {{
  $Clauses = $Selector -split '&&'
  foreach ($Clause in $Clauses) {{
    $Trimmed = $Clause.Trim()
    if ([string]::IsNullOrWhiteSpace($Trimmed)) {{ continue }}
    if (-not (Matches-Clause $Element $Trimmed)) {{ return $false }}
  }}
  return $true
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

function Find-WindowAncestor($Element) {{
  $Walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
  $CurrentElement = $Element
  while ($null -ne $CurrentElement) {{
    try {{
      if ($CurrentElement.Current.ControlType -eq
          [System.Windows.Automation.ControlType]::Window) {{
        return $CurrentElement
      }}
      $CurrentElement = $Walker.GetParent($CurrentElement)
    }} catch {{ return $null }}
  }}
  return $null
}}

function Get-PointValue($Point, [string]$Name, [int]$Index) {{
  if ($null -ne $Point.PSObject.Properties[$Name]) {{
    return [double]$Point.$Name
  }}
  return [double]$Point[$Index]
}}

function Resolve-Coordinate([double]$Raw, [double]$Origin, [double]$Size) {{
  if ([Math]::Abs($Raw) -le 1.0) {{
    return [int][Math]::Round($Origin + ($Raw * $Size))
  }}
  return [int][Math]::Round($Raw)
}}

function Invoke-DrawPath($Element, [string]$PathJson) {{
  $Rect = $Element.Current.BoundingRectangle
  if ($Rect.Width -le 0 -or $Rect.Height -le 0) {{
    throw "Cannot draw on a target with empty bounds"
  }}
  $Parsed = $PathJson | ConvertFrom-Json
  if ($null -ne $Parsed.PSObject.Properties['points']) {{
    $Points = @($Parsed.points)
  }} else {{
    $Points = @($Parsed)
  }}
  if ($Points.Count -lt 2) {{ throw "draw_path requires at least two points" }}

  $Resolved = New-Object System.Collections.Generic.List[object]
  foreach ($Point in $Points) {{
    $RawX = Get-PointValue $Point 'x' 0
    $RawY = Get-PointValue $Point 'y' 1
    $X = Resolve-Coordinate $RawX $Rect.X $Rect.Width
    $Y = Resolve-Coordinate $RawY $Rect.Y $Rect.Height
    $Resolved.Add([pscustomobject]@{{ x = $X; y = $Y }})
  }}

  [AgentOSMouse]::SetCursorPos($Resolved[0].x, $Resolved[0].y) | Out-Null
  [AgentOSMouse]::mouse_event(0x0002, 0, 0, 0, 0)
  Start-Sleep -Milliseconds 40
  $CurrentX = $Resolved[0].x
  $CurrentY = $Resolved[0].y
  for ($Index = 1; $Index -lt $Resolved.Count; $Index += 1) {{
    $Previous = $Resolved[$Index - 1]
    $Point = $Resolved[$Index]
    $Dx = $Point.x - $Previous.x
    $Dy = $Point.y - $Previous.y
    $Steps = [Math]::Max(1, [int][Math]::Ceiling(
      [Math]::Max([Math]::Abs($Dx), [Math]::Abs($Dy)) / 18.0
    ))
    for ($Step = 1; $Step -le $Steps; $Step += 1) {{
      $X = [int][Math]::Round($Previous.x + (($Dx * $Step) / $Steps))
      $Y = [int][Math]::Round($Previous.y + (($Dy * $Step) / $Steps))
      [AgentOSMouse]::mouse_event(0x0001, $X - $CurrentX, $Y - $CurrentY, 0, 0)
      $CurrentX = $X
      $CurrentY = $Y
      Start-Sleep -Milliseconds 8
    }}
  }}
  [AgentOSMouse]::mouse_event(0x0004, 0, 0, 0, 0)
  return $Resolved
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
if ($ActionType -eq 'draw_path' -and $null -ne $FocusError) {{
  $WindowTarget = Find-WindowAncestor $Target
  if ($null -ne $WindowTarget) {{
    try {{ $WindowTarget.SetFocus() }} catch {{ }}
  }}
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
$DrawPath = $null
if ($ActionType -eq 'draw_path') {{
  $DrawPath = Invoke-DrawPath $Target $Value
  $Status = 'drawn'
}}
[pscustomobject]@{{
  status = $Status
  action_type = $ActionType
  selector = $Selector
  matched_name = $Target.Current.Name
  matched_role = $Target.Current.ControlType.ProgrammaticName
  focus_error = $FocusError
  draw_path = $DrawPath
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
