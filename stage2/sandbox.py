"""阶段 2 · 代码执行沙箱：本地受限子进程 + Docker 两种后端。

先明确一件事：**本地子进程不是真正的沙箱**。它能读你的文件、能联网、能跑满 CPU。
它只做到三件事：独立临时目录、超时、清洗环境变量。真正的隔离靠 Docker/E2B。

运行演示：uv run python stage2/sandbox.py
Docker 模式：SANDBOX_BACKEND=docker uv run python stage2/sandbox.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass(frozen=True)
class SandboxResult:
    backend: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    note: str = ""

    def as_text(self) -> str:
        parts = [
            f"[{self.backend}] {'成功' if self.ok else '失败'}，"
            f"退出码 {self.exit_code}，耗时 {self.duration:.2f}s"
        ]
        if self.note:
            parts.append(self.note)
        if self.stdout:
            parts.append("stdout:\n" + self.stdout[:4000])
        if self.stderr:
            parts.append("stderr:\n" + self.stderr[:2000])
        return "\n".join(parts)


def _clean_env() -> dict[str, str]:
    """复制环境变量，但删掉所有像密钥的东西 —— 别把 .env 泄给用户代码。"""
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SECRET_MARKERS)
    }
    env["PYTHONUTF8"] = "1"
    return env


class LocalSandbox:
    """受限子进程：临时目录 + 超时 + 清洗环境变量。不是安全边界。"""

    name = "local"

    def run(self, code: str, timeout: float = 5.0) -> SandboxResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="agent_sbx_") as tmp:
            script = Path(tmp) / "main.py"
            script.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-X", "utf8", str(script)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    env=_clean_env(),
                )
                return SandboxResult(
                    backend=self.name,
                    ok=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    duration=time.perf_counter() - started,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(
                    backend=self.name,
                    ok=False,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration=time.perf_counter() - started,
                    note=f"超时（>{timeout}s），子进程已被杀死。",
                )
            except OSError as exc:
                return SandboxResult(
                    backend=self.name,
                    ok=False,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration=time.perf_counter() - started,
                    note=f"无法启动子进程：{exc}",
                )


class DockerSandbox:
    """容器沙箱：断网、只读根文件系统、内存/CPU/进程数限制、无特权。"""

    name = "docker"

    def __init__(self, image: str = "python:3.13-alpine") -> None:
        self.image = image

    def available(self) -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            proc = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                timeout=5,
            )
            return proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def run(self, code: str, timeout: float = 10.0) -> SandboxResult:
        started = time.perf_counter()
        name = f"agent-sbx-{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory(prefix="agent_sbx_") as tmp:
            script = Path(tmp) / "main.py"
            script.write_text(code, encoding="utf-8")
            volume = f"{Path(tmp).resolve()}:/work:ro"
            cmd = [
                "docker", "run", "--rm", "--name", name,
                "--network", "none",
                "--memory", "256m",
                "--cpus", "0.5",
                "--pids-limit", "64",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--read-only",
                "--tmpfs", "/tmp:size=16m",
                "-v", volume,
                "-w", "/work",
                self.image,
                "python", "-I", "-X", "utf8", "/work/main.py",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    env=_clean_env(),
                )
                return SandboxResult(
                    backend=self.name,
                    ok=proc.returncode == 0,
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    duration=time.perf_counter() - started,
                )
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["docker", "rm", "-f", name], capture_output=True, timeout=15
                )
                return SandboxResult(
                    backend=self.name,
                    ok=False,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration=time.perf_counter() - started,
                    note=f"容器超时（>{timeout}s），已强制删除。",
                )
            except OSError as exc:
                return SandboxResult(
                    backend=self.name,
                    ok=False,
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration=time.perf_counter() - started,
                    note=f"无法启动 Docker：{exc}",
                )


def get_sandbox(backend: str | None = None) -> LocalSandbox | DockerSandbox:
    name = (backend or os.getenv("SANDBOX_BACKEND") or "local").lower()
    if name == "docker":
        sandbox = DockerSandbox()
        if not sandbox.available():
            raise RuntimeError(
                "Docker 不可用：请启动 Docker Desktop，或设置 SANDBOX_BACKEND=local"
            )
        return sandbox
    return LocalSandbox()


def run_code(
    code: str, backend: str | None = None, timeout: float | None = None
) -> SandboxResult:
    try:
        sandbox = get_sandbox(backend)
    except RuntimeError as exc:
        return SandboxResult(
            backend=backend or "unknown",
            ok=False,
            exit_code=None,
            stdout="",
            stderr="",
            duration=0.0,
            note=str(exc),
        )
    if timeout is None:
        timeout = 10.0 if sandbox.name == "docker" else 5.0
    return sandbox.run(code, timeout)


if __name__ == "__main__":
    DEMOS = [
        ("正常计算", "print(sum(range(1, 101)))"),
        (
            "看看环境变量里的密钥还在不在",
            "import os\nprint('DEEPSEEK_API_KEY =', os.environ.get('DEEPSEEK_API_KEY'))",
        ),
        ("无限循环（会被超时杀掉）", "while True:\n    pass"),
    ]
    for title, code in DEMOS:
        print("=" * 60)
        print(title)
        print(run_code(code, timeout=1.5).as_text())
