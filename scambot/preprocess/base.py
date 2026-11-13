"""Shared pieces for the media extraction engines."""

import asyncio
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Final, Literal

ToolErrorCode = Literal["unavailable", "timeout", "failed", "no_audio", "bad_image"]

# Only local files: a crafted playlist cannot make ffmpeg fetch URLs.
FFMPEG_INPUT_FLAGS: Final = (
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-protocol_whitelist",
    "file",
)


@dataclass(frozen=True)
class Extracted:
    """Text read from a file and, when known, its language."""

    text: str
    language: str | None


class MediaToolError(Exception):
    """An extraction tool is missing or failed; ``code`` says which way."""

    def __init__(self, code: ToolErrorCode) -> None:
        super().__init__(code)
        self.code: ToolErrorCode = code


def run_tool(args: Sequence[str], timeout_s: float) -> bytes:
    """Run an external program and return its output; stderr is never kept in messages."""
    try:
        done = subprocess.run(  # noqa: S603 - absolute binary path, fixed arguments
            list(args), capture_output=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        raise MediaToolError("timeout") from None
    except FileNotFoundError:
        raise MediaToolError("unavailable") from None
    if done.returncode != 0:
        if b"does not contain any stream" in done.stderr:
            raise MediaToolError("no_audio")
        raise MediaToolError("failed")
    return done.stdout


async def run_blocking[T](pool: Executor, fn: Callable[[], T], timeout_s: float) -> T:
    """Run blocking ``fn`` in ``pool``; the caller stops waiting after ``timeout_s``.

    The thread itself cannot be killed, so ``fn`` must bound its own run time.
    """
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(loop.run_in_executor(pool, fn), timeout_s)
