"""Write stdout and stderr as UTF-8 when they are redirected.

On Windows, Python writes redirected output (a file or a pipe) in the legacy code page, such as
cp1252. Printing a beatmap title in Japanese or another non-Latin script then raises
UnicodeEncodeError and stops the run. That happens when output is saved to a log file, and in
MaiMod's web UI, which reads mai_mod.py through a pipe. Consoles already use UTF-8 and are left
alone.

Importing this module applies the change.
"""
import sys


def use_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding == "utf8" or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


use_utf8_output()
