"""
progress.py
===========
A small helper that prints how far a long loop has got, at most once every
N seconds, together with elapsed time and an estimate of the time remaining
(ETA). This satisfies the requirement that long-running code reports progress.

How to use:
-----------
    from common.progress import ProgressPrinter

    pp = ProgressPrinter(total=1000, name="Simulating UEs", print_every_seconds=30)
    for i in range(1000):
        ...do work...
        pp.update(i + 1)      # tell it how many steps are done
    pp.done()

It only prints when at least `print_every_seconds` have passed since the last
print, so it will not flood the console.
"""

import time


def _format_seconds(seconds: float) -> str:
    """Turn a number of seconds into a short 'HH:MM:SS' style string."""
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class ProgressPrinter:
    """Prints progress of a loop with percentage, elapsed time, and ETA."""

    def __init__(self, total: int, name: str = "Progress",
                 print_every_seconds: float = 30.0, show_eta: bool = True):
        self.total = max(1, int(total))     # total number of steps
        self.name = name
        self.print_every = float(print_every_seconds)
        self.show_eta = show_eta

        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_count = 0
        # Always print a starting line so the user knows work has begun.
        print(f"[{self.name}] started — {self.total} steps to do.", flush=True)

    def update(self, count: int) -> None:
        """Call this with the number of steps completed so far."""
        now = time.time()
        # Only print if enough time has passed, or if we just finished.
        if (now - self.last_print_time) < self.print_every and count < self.total:
            return

        elapsed = now - self.start_time
        fraction = count / self.total
        percent = 100.0 * fraction

        # Estimate time remaining from the average rate so far.
        if count > 0:
            rate = count / elapsed                    # steps per second
            remaining = (self.total - count) / rate if rate > 0 else 0.0
        else:
            remaining = 0.0

        line = (f"[{self.name}] {percent:5.1f}%  "
                f"({count}/{self.total})  "
                f"elapsed {_format_seconds(elapsed)}")
        if self.show_eta:
            line += f"  ETA {_format_seconds(remaining)}"
        print(line, flush=True)

        self.last_print_time = now
        self.last_count = count

    def done(self) -> None:
        """Print a final line with total time taken."""
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] done — 100.0% in {_format_seconds(elapsed)}.",
              flush=True)
