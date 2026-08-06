import io

from inflorescence.progress import NullProgress, StderrProgress, resolve


def test_resolve_returns_null_progress_for_none() -> None:
    reporter = resolve(None)
    assert isinstance(reporter, NullProgress)
    # NullProgress.update is a no-op and must never raise.
    reporter.update("parse", 1, 10)


def test_resolve_passes_through_a_real_reporter() -> None:
    bar = StderrProgress(stream=io.StringIO())
    assert resolve(bar) is bar


def test_stderr_progress_renders_percent_and_closes_line_at_total() -> None:
    stream = io.StringIO()
    bar = StderrProgress(stream=stream, width=10)

    bar.update("parse", 5, 10)
    bar.update("parse", 10, 10)

    out = stream.getvalue()
    assert "parse" in out
    assert " 50% (5/10)" in out
    assert "100% (10/10)" in out
    assert out.endswith("\n")  # line closed on completion


def test_stderr_progress_newlines_between_phases() -> None:
    stream = io.StringIO()
    bar = StderrProgress(stream=stream, width=10)

    bar.update("parse", 5, 10)      # line left open (not at total)
    bar.update("store", 1, 2)       # phase switch must finalize the previous line

    out = stream.getvalue()
    assert "parse" in out and "store" in out
    assert "\n" in out  # a newline separates the two phase lines


def test_stderr_progress_ignores_zero_total() -> None:
    stream = io.StringIO()
    bar = StderrProgress(stream=stream)
    bar.update("parse", 0, 0)
    assert stream.getvalue() == ""
