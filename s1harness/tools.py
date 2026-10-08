import subprocess
from dataclasses import dataclass
from pathlib import Path

from .pipeline import BASH_QUESTION, BASH_TIMEOUT, BASH_WITH_QUESTION, QUESTION_REQUIRED, READ_WITH_QUESTION

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def schemas(relevance: bool = False) -> list[dict]:
    """the tool catalog. with the relevance filter on, read and bash take a required question"""
    tools = [
        _fn("read", "Read a file. Lines come back numbered. Optionally read only a range of lines.",
            {"path": {"type": "string"},
             "start_line": {"type": "integer", "description": "First line to read, from 1."},
             "end_line": {"type": "integer", "description": "Last line to read, inclusive."}},
            ["path"]),
        _fn("list", "List one directory, one level deep. Directories end with a slash.",
            {"path": {"type": "string", "description": "Directory to list. Use . for the working directory."}},
            ["path"]),
        _fn("edit", "Replace one exact occurrence of old_text with new_text in a file. "
            "Fails if old_text matches zero times or more than once.",
            {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            ["path", "old_text", "new_text"]),
        _fn("write", "Create a file, or overwrite it, with the given content.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"]),
        _fn("bash", "Run one shell command in the working directory. Use read and list to look at files.",
            {"command": {"type": "string"}},
            ["command"]),
    ]
    if relevance:
        for t in tools:
            f = t["function"]
            if f["name"] == "read":
                f["description"] = READ_WITH_QUESTION
                f["parameters"]["properties"]["question"] = dict(QUESTION_REQUIRED)
            elif f["name"] == "bash":
                f["description"] += BASH_WITH_QUESTION
                f["parameters"]["properties"]["question"] = dict(BASH_QUESTION)
            else:
                continue
            f["parameters"]["required"] = [*f["parameters"]["required"], "question"]
    return tools


class ToolError(Exception):
    pass


@dataclass
class Result:
    text: str  # a read's numbered lines, bash's output without the exit code, or any other tool's reply
    ok: bool = True
    exit_code: int | None = None
    lines: list[tuple[int, str]] | None = None  # a read's (file line number, text) per line
    path: str | None = None
    total: int | None = None  # a read's line count in the whole file
    file_lines: list[str] | None = None  # every line of the file a read read, for a python file's imports header


@dataclass
class Tools:
    workdir: Path

    def _resolve(self, rel: str) -> Path:
        p = (self.workdir / rel).resolve()
        if not p.is_relative_to(self.workdir.resolve()):
            raise ToolError(f"{rel} is outside the working directory")
        return p

    def run(self, name: str, args: dict) -> Result:
        try:
            return getattr(self, f"_{name}")(**args)
        except ToolError as e:
            return Result(f"Error: {e}", ok=False, path=args.get("path"))
        except TypeError as e:
            return Result(f"Error: bad arguments: {e}", ok=False, path=args.get("path"))
        except (OSError, UnicodeDecodeError) as e:
            return Result(f"Error: {e}", ok=False, path=args.get("path"))

    def _read(self, path: str, start_line: int | None = None, end_line: int | None = None) -> Result:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        text = p.read_text(errors="replace").splitlines()
        first = max(start_line or 1, 1)
        last = min(end_line or len(text), len(text))
        lines = [(n, text[n - 1]) for n in range(first, last + 1)]
        body = "\n".join(f"{n:>5}\t{t}" for n, t in lines) or "(empty)"
        return Result(body, lines=lines, path=path, total=len(text), file_lines=text)

    def _list(self, path: str = ".") -> Result:
        p = self._resolve(path)
        if not p.is_dir():
            raise ToolError(f"{path} is not a directory")
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir() if c.name not in SKIP_DIRS)
        return Result("\n".join(names) or "(empty)", path=path)

    def _edit(self, path: str, old_text: str, new_text: str) -> Result:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        text = p.read_text()
        n = text.count(old_text)
        if n != 1:
            raise ToolError(f"old_text matches {n} times in {path}, it must match exactly once")
        p.write_text(text.replace(old_text, new_text))
        return Result(f"Edited {path}.", path=path)

    def _write(self, path: str, content: str) -> Result:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return Result(f"Wrote {path}.", path=path)

    def _bash(self, command: str) -> Result:
        """the whole output, no cut: the filters see all of it. the exit code comes back apart, and the loop adds
        its line after the filters"""
        try:
            r = subprocess.run(command, shell=True, cwd=self.workdir, capture_output=True,
                               text=True, timeout=BASH_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise ToolError(f"timed out after {BASH_TIMEOUT} s")
        return Result((r.stdout + r.stderr).strip(), exit_code=r.returncode)
