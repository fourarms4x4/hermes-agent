"""Idle screensaver - Rick-on-a-bike animation.

Renders when agent is busy AND no keyboard input for configured idle_seconds.
Dismisses on any keypress or when agent turn completes.

Canonical implementation — the standalone script
(~/.hermes/screensaver/screensaver.py, symlinked to ~/.local/bin/bike-screensaver)
is a thin shim that imports this module.

Design notes (post-Claude-consult 2026-08-10):
- ALT-SCREEN BUFFER (\x1b[?1049h / \x1b[?1049l): the user's terminal content is
  restored untouched on dismissal. NEVER \x1b[2J the main screen — it wipes the
  conversation.
- Terminal-size-aware nearest-neighbor scaling. Source frames are 200x100
  (bicubic-extracted from the 1080p clip); each frame is scaled once per
  activation to cols x (2*rows), capped at MAX_CELLS to keep ANSI throughput
  sane on slow/remote terminals.
- Constant-rate pacing: sleep the full frame interval every frame. Never skip
  sleep to "catch up" — dt-based catch-up caused visible fast-forward.
- Row-diff redraw: only rows that changed are re-emitted, with an explicit
  row-home escape. Cuts output massively for static background regions.
- Thread lifecycle: start() refuses to spawn over a live zombie thread (a
  timed-out join leaves the reference in place); the animation loop lives
  exactly as long as _running is True; stop() sets _running False so the
  thread exits at the next row boundary (checked between row writes).
- Writes are row-sized chunks, so a stop() lands within ~1 row of output —
  no giant single-write frames that can interleave with prompt_toolkit.
"""
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

DEFAULT_ASSET = str(Path.home() / ".hermes/screensaver/bike/bike.raw")
MAX_CELLS = 72000  # ~360x200 half-block cells; cap for slow/remote terminals


def render_frame(frame: bytes, width: int, height: int):
    """width x height RGB24 frame -> height/2 rows of half-block truecolor ANSI.

    Returns a LIST of row strings (row-diff redraw relies on that).
    """
    rows = []
    for y in range(0, height, 2):
        cells = []
        for x in range(width):
            ti = (y * width + x) * 3
            bi = ((y + 1) * width + x) * 3 if y + 1 < height else ti
            tr, tg, tb = frame[ti], frame[ti + 1], frame[ti + 2]
            br, bg, bb = frame[bi], frame[bi + 1], frame[bi + 2]
            cells.append(
                f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀"
            )
        rows.append("".join(cells) + "\x1b[0m")
    return rows


def _index_maps(sw: int, sh: int, dw: int, dh: int):
    """Nearest-neighbor source index maps for scaling sw*sh -> dw*dh."""
    xs = [min((x * sw) // dw, sw - 1) for x in range(dw)]
    ys = [min((y * sh) // dh, sh - 1) for y in range(dh)]
    return xs, ys


def scale_frame(frame: bytes, sw: int, sh: int, dw: int, dh: int, xs, ys) -> bytes:
    """Nearest-neighbor scale one RGB24 frame. Pure Python, no deps."""
    if (sw, sh) == (dw, dh):
        return frame
    out = bytearray(dw * dh * 3)
    o = 0
    for y in ys:
        row = y * sw * 3
        for x in xs:
            s = row + x * 3
            out[o] = frame[s]
            out[o + 1] = frame[s + 1]
            out[o + 2] = frame[s + 2]
            o += 3
    return bytes(out)


def ensure_frames(asset_path: Path) -> None:
    """Extract RGB24 frames from the seamless loop clip if bike.raw is missing.

    Prefers bike-loop.mp4 (16s crossfade loop — no jump cut at the wrap);
    falls back to bike.mp4.
    """
    if asset_path.exists():
        return
    print("Extracting frames (one-time)...", file=sys.stderr)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    src = asset_path.parent.parent / "bike-loop.mp4"
    if not src.exists():
        src = asset_path.parent.parent / "bike.mp4"
    if not src.exists():
        sys.exit(f"missing {src} — drop the clip at that path first")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-vf", "fps=12,scale=200:100:flags=bicubic", "-pix_fmt", "rgb24",
         "-f", "rawvideo", str(asset_path)],
        check=True,
    )
    n = asset_path.stat().st_size // (200 * 100 * 3)
    (asset_path.parent / "manifest.json").write_text(json.dumps({
        "fps": 12, "width": 200, "height": 100, "frame_count": n,
    }))


class Screensaver:
    """Idle screensaver manager.

    Monitors keyboard activity and agent busy state. When both conditions
    are met (idle > threshold AND agent busy), renders animation frames in
    the alternate screen buffer. Any keypress dismisses (via the CLI's
    catch-all key binding → record_keypress), or the turn completing.
    """

    def __init__(self, config: dict, print_fn=None):
        """Initialize screensaver.

        Args:
            config: display.screensaver config dict with keys:
                - enabled (bool): whether screensaver is active
                - idle_seconds (int): idle threshold before activation
                - asset_path (str): path to bike.raw frames
            print_fn: callable to route output through (e.g. _pt_print wrapper)
        """
        self.enabled = config.get("enabled", True)
        self.idle_seconds = config.get("idle_seconds", 10)
        self.asset_path = Path(
            config.get("asset_path", DEFAULT_ASSET)
        ).expanduser()
        self._print_fn = print_fn

        # State tracking
        self._last_keypress = time.monotonic()
        self._busy = False
        self._running = False
        self._thread = None
        self._dismissed = False
        self._on_screen = False
        self._last_rows = None
        self._frame_pos = 0  # persists across pause/resume so playback continues

        # Frame data (loaded lazily)
        self._frames = []
        self._fps = 12
        self._width = 200
        self._height = 100
        self._loaded = False

        # Terminal state
        self._out = sys.stdout
        self._is_tty = hasattr(self._out, 'isatty') and self._out.isatty()

    # ------------------------------------------------------------------ #
    # Public API used by the CLI                                          #
    # ------------------------------------------------------------------ #

    def is_active(self) -> bool:
        """True while the animation is on screen (drives the any-key filter)."""
        return self._on_screen

    def record_keypress(self):
        """Record keyboard activity - dismisses immediately."""
        self._last_keypress = time.monotonic()
        self._dismissed = True

    def reset_idle(self):
        """Reset idle timer without dismissing (used when user submits input)."""
        self._last_keypress = time.monotonic()

    def set_busy(self, busy: bool):
        """Set agent busy state.

        Called when thinking starts/stops. ``busy=True`` (thinking started)
        arms the screensaver; ``busy=False`` (thinking paused mid-turn, e.g.
        before a tool call) only PAUSES — the thread stays alive and the
        animation resumes in place on the next thinking pulse, so playback
        is continuous for the whole turn. Full teardown happens at turn end
        via ``stop()`` (called by the CLI when the agent run completes).
        """
        self._busy = busy
        if busy:
            # New busy period - allow screensaver to potentially activate again
            self._dismissed = False
        else:
            # Mid-turn pause: _busy False makes should_activate() False, so
            # the animation thread leaves the alt screen at its next frame
            # boundary. Thread and frame position are preserved.
            pass

    def pause(self):
        """Alias for the mid-turn pause (thinking stopped, tool output next)."""
        self._busy = False

    def start(self):
        """Start screensaver monitoring."""
        if not self.enabled or self._running:
            return
        if self._thread and self._thread.is_alive():
            # Zombie from a timed-out join — never double-write stdout
            return
        self.load_frames()
        if not self._loaded or not self._is_tty:
            return
        self._running = True
        self._dismissed = False
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop screensaver and clean up."""
        self._running = False
        self._dismissed = False
        if self._thread:
            self._thread.join(timeout=0.5)
            if not self._thread.is_alive():
                self._thread = None
            # If still alive, KEEP the reference: start() refuses while it
            # lives, so a zombie can never spawn a second writer.

    # ------------------------------------------------------------------ #
    # Frame loading                                                       #
    # ------------------------------------------------------------------ #

    def load_frames(self):
        """Load frame data from raw file."""
        if self._loaded:
            return
        if not self.asset_path.exists():
            return
        try:
            manifest_path = self.asset_path.parent / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                self._fps = manifest.get("fps", 12)
                self._width = manifest.get("width", 200)
                self._height = manifest.get("height", 100)
            raw = self.asset_path.read_bytes()
            frame_size = self._width * self._height * 3
            num_frames = len(raw) // frame_size
            self._frames = [
                raw[i * frame_size:(i + 1) * frame_size]
                for i in range(num_frames)
            ]
            self._loaded = True
        except Exception:
            self._frames = []
            self._loaded = False

    # ------------------------------------------------------------------ #
    # Activation logic                                                    #
    # ------------------------------------------------------------------ #

    def should_activate(self) -> bool:
        """Check if screensaver should activate."""
        if not self.enabled or not self._loaded or not self._is_tty:
            return False
        idle_time = time.monotonic() - self._last_keypress
        return self._busy and idle_time >= self.idle_seconds and not self._dismissed

    def _target_size(self):
        """Frame size matching the current terminal (half-block: 2px per row)."""
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 80, 40
        cols = max(cols, 20)
        rows = max(rows, 5)
        dw, dh = cols, rows * 2
        while dw * dh > MAX_CELLS:
            dw = int(dw * 0.9)
            dh = int(dh * 0.9)
        return dw, dh

    # ------------------------------------------------------------------ #
    # Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _write(self, text: str):
        """Write to stdout or via print_fn."""
        if self._print_fn is not None:
            try:
                self._print_fn(text)
            except Exception:
                pass
            return
        try:
            self._out.write(text)
            self._out.flush()
        except (ValueError, OSError):
            pass

    def _draw(self, frame: bytes, width: int, height: int):
        """Row-diff redraw: emit only changed rows; abort if _running flips.

        Each emitted row carries an explicit row-home escape, so skipped rows
        need no cursor bookkeeping.
        """
        rows = render_frame(frame, width, height)
        for r, row in enumerate(rows):
            if not self._running:
                return
            if row == self._last_rows[r]:
                continue
            self._last_rows[r] = row
            self._write(f"\x1b[{r + 1};1H{row}")

    def _animate(self, *, watch_keys: bool = False):
        """Animation loop - runs in background thread (or foreground standalone)."""
        self.load_frames()
        if not self._frames:
            return

        while self._running and not self._dismissed:
            if not self.should_activate():
                time.sleep(0.1)
                continue

            # Scale frames lazily to the current terminal size (first frame
            # renders ~instantly; the rest fill the cache as they're drawn)
            dw, dh = self._target_size()
            xs, ys = _index_maps(self._width, self._height, dw, dh)
            scaled = [None] * len(self._frames)

            def get_frame(i: int) -> bytes:
                f = scaled[i]
                if f is None:
                    f = scale_frame(self._frames[i], self._width, self._height, dw, dh, xs, ys)
                    scaled[i] = f
                return f

            frame_interval = 1.0 / self._fps
            self._last_rows = [None] * (dh // 2)

            # Enter alternate screen buffer, hide cursor, clear (alt only)
            self._on_screen = True
            self._write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")

            try:
                frame_idx = self._frame_pos % len(scaled)
                next_t = time.monotonic() + frame_interval
                while self._running and not self._dismissed:
                    if not self.should_activate():
                        break
                    self._draw(get_frame(frame_idx), dw, dh)
                    frame_idx = (frame_idx + 1) % len(scaled)
                    # Deadline-based pacing: sleep the REMAINING time until the
                    # next frame slot. Robust against early sleep returns
                    # (EINTR) — re-sleeps the remainder instead of drifting
                    # fast; and never bursts ahead of the timeline.
                    while True:
                        dt = next_t - time.monotonic()
                        if dt <= 0:
                            break
                        time.sleep(dt)
                    next_t += frame_interval
                    if watch_keys:
                        try:
                            if select.select([sys.stdin], [], [], 0)[0]:
                                self._dismissed = True
                                break
                        except (ValueError, OSError):
                            pass
            finally:
                # Leave alternate screen buffer — original content restored
                self._write("\x1b[0m\x1b[?25h\x1b[?1049l")
                self._on_screen = False
                self._last_rows = None
                self._frame_pos = frame_idx  # resume here after next pause

            if watch_keys:
                break  # standalone: one activation cycle then exit

    # ------------------------------------------------------------------ #
    # Standalone mode                                                     #
    # ------------------------------------------------------------------ #

    def run_foreground(self):
        """Standalone screensaver: render immediately, any key quits."""
        ensure_frames(self.asset_path)
        self.load_frames()
        if not self._frames:
            return

        fd = sys.stdin.fileno()
        old = None
        try:
            import termios
            import tty
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            old = None  # not a tty — still render, quit via Ctrl-C/EOF

        # Standalone mode: the user asked for it — render regardless of
        # stdout tty-ness (integration mode keeps the isatty gate).
        self._is_tty = True
        # Activate immediately regardless of idle_seconds/busy state
        self._busy = True
        self._last_keypress = time.monotonic() - max(self.idle_seconds, 1) - 1
        self._running = True
        self._dismissed = False

        try:
            self._animate(watch_keys=True)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._write("\x1b[0m\x1b[?25h\x1b[?1049l")
            if old is not None:
                import termios
                termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    Screensaver({"enabled": True, "idle_seconds": 10}).run_foreground()
