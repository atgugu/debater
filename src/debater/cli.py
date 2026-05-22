"""CLI entry point: argument parsing, signal handling, server boot."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from http.server import ThreadingHTTPServer

from debater.registry import DebateRegistry
from debater.server import make_handler

log = logging.getLogger("debater")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent debate system")
    parser.add_argument("--port", type=int, default=8081, help="HTTP port (default: 8081)")
    parser.add_argument("--debates-dir", default="debates", help="Debates directory (default: debates)")
    parser.add_argument("--max-concurrent-claude", type=int, default=2,
                        help="Max concurrent Claude calls (default: 2)")
    args = parser.parse_args()

    os.makedirs(args.debates_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    registry = DebateRegistry(args.debates_dir, args.max_concurrent_claude)

    def handle_signal(signum, frame):
        registry.shutdown_flag = True
        registry.signal_all_stop()
        if registry.server:
            threading.Thread(target=registry.server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    handler_cls = make_handler(registry)
    server = ThreadingHTTPServer(("", args.port), handler_cls)
    server.allow_reuse_address = True
    registry.server = server

    log.info("Debater running at http://localhost:%d", args.port)
    print(f"Debater running at http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down...")
        registry.signal_all_stop()
        registry.join_all(timeout=5)
        server.server_close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
