# Contributing

## Dev setup

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Module layout

The package is intentionally split so each module has one responsibility:

- **`config.py`** — pure constants. No imports beyond stdlib.
- **`registry.py`** — owns all shared runtime state: locks, thread bookkeeping, the Claude semaphore, the HTTP server reference. A single `DebateRegistry` instance is created in `cli.main()` and threaded through every code path that needs runtime state.
- **`state.py`** — disk I/O for `state.json` and debate directories. Pure with respect to runtime state; only knows about the filesystem.
- **`claude_runner.py`** — the *only* module that spawns subprocesses. All other modules call `run_agent(registry, ...)`.
- **`prompts.py`** — pure prompt templates. No side effects, no imports from `claude_runner`, `state`, or `orchestrator`.
- **`research.py`** — research-request extraction + the recursive "run agent, intercept research tags, re-run agent" loop.
- **`orchestrator.py`** — the round cycle and full-debate driver. The only module that calls the prompt builders + the claude runner + state mutators in sequence.
- **`server.py`** — HTTP routes. `make_handler(registry)` returns a request handler class with the registry bound as a class attribute.
- **`cli.py`** — `main()`, argparse, signal handler. Constructs the registry and starts the server.

## Lock-nesting invariant

**Registry-wide lock (`registry.lock`) is always acquired BEFORE any per-debate lock (`registry.state_lock(id)`).** Never invert this ordering or you will deadlock with the signal handler.

In practice this means: code paths that need both should grab `registry.lock` first via one of the registry's helper methods, then exit that critical section before acquiring a `state_lock`. Most code uses the helper methods on `DebateRegistry` (e.g. `register_debate`, `kill_process`) and never touches `registry.lock` directly.

## Testing

The test suite mocks `claude_runner.run_agent` so no Claude CLI is required to run tests. See `tests/conftest.py` for shared fixtures.

## Code style

- `ruff check` and `ruff format` are the source of truth.
- Line length 110.
- Prefer explicit `registry: DebateRegistry` parameters over global state. The whole point of the refactor was killing globals — please don't reintroduce them.
