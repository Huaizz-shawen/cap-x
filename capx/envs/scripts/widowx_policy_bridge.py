#!/usr/bin/env python3
"""Bridge WidowX real-robot msgpack loop with an HTTP policy server.

Data flow:
1) Robot client sends observation to this bridge via msgpack TCP.
2) Bridge POSTs {"observation": ..., "instruction": ...} to policy /act endpoint.
3) Bridge writes policy response back as latest_action for robot client to consume.

This is intentionally lightweight so it can be adapted to robot-specific command schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from typing import Any

import numpy as np
import requests

from capx.utils.msgpack_server_client_utils import MsgpackNumpyServer

LOG = logging.getLogger("widowx_policy_bridge")


def _start_msgpack_server_in_background(server: MsgpackNumpyServer) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return loop, t


def _to_jsonable(obj: Any) -> Any:
    """Convert msgpack/numpy-rich payloads into JSON-safe Python objects."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, bytes):
                key = k.decode("utf-8", errors="replace")
            else:
                key = str(k)
            out[key] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()
    return obj


def _extract_action_vector(policy_response: dict[str, Any], robot_name: str) -> list[float] | None:
    actions = policy_response.get("actions")
    if not isinstance(actions, dict):
        return None
    vec = actions.get(robot_name)
    if isinstance(vec, list):
        return vec
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="WidowX policy bridge (msgpack <-> HTTP)")
    parser.add_argument("--bridge-host", default="0.0.0.0", help="Host for msgpack bridge server")
    parser.add_argument("--bridge-port", type=int, default=9000, help="Port for msgpack bridge server")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8000/act", help="Policy /act endpoint")
    parser.add_argument("--instruction", default=None, help="Optional instruction sent with each /act request")
    parser.add_argument("--robot-name", default="arm", help="Key in policy response actions dict, e.g. arm or wx250s")
    parser.add_argument("--request-timeout", type=float, default=3.0, help="HTTP timeout in seconds")
    parser.add_argument("--poll-hz", type=float, default=20.0, help="Polling frequency for new observations")
    parser.add_argument(
        "--publish-raw-response",
        action="store_true",
        help="Publish full policy JSON response as latest_action instead of normalized command",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = MsgpackNumpyServer(host=args.bridge_host, port=args.bridge_port)
    _loop, _thread = _start_msgpack_server_in_background(server)

    LOG.info("Bridge listening on %s:%d", args.bridge_host, args.bridge_port)
    LOG.info("Forwarding observations to %s", args.policy_url)

    period = 1.0 / max(args.poll_hz, 1e-6)
    last_obs_signature: tuple[int, int] | None = None

    while True:
        obs = server.latest_observation
        if obs is None:
            time.sleep(period)
            continue

        # Heuristic signature to avoid re-querying policy when observation did not update.
        obs_signature = (id(obs), len(obs))
        if obs_signature == last_obs_signature:
            time.sleep(period)
            continue
        last_obs_signature = obs_signature

        payload = {
            "observation": _to_jsonable(obs),
            "instruction": args.instruction,
        }

        try:
            resp = requests.post(args.policy_url, json=payload, timeout=args.request_timeout)
            resp.raise_for_status()
            policy_data = resp.json()

            if args.publish_raw_response:
                server.latest_action = policy_data
            else:
                action_vec = _extract_action_vector(policy_data, args.robot_name)
                if action_vec is None:
                    LOG.warning(
                        "Policy response missing actions[%r], fallback to raw response keys=%s",
                        args.robot_name,
                        sorted(policy_data.keys()),
                    )
                    server.latest_action = policy_data
                else:
                    # Normalized command payload for robot clients that expect simple action schema.
                    server.latest_action = {
                        "timestamp": time.time(),
                        "mode": "policy",
                        args.robot_name: action_vec,
                        "meta": {
                            "policy_step": policy_data.get("step"),
                            "target_position": policy_data.get("target_position"),
                        },
                    }

            LOG.info("Forwarded one observation -> action. response=%s", json.dumps(policy_data)[:300])
        except requests.RequestException as exc:
            LOG.exception("Policy request failed: %s", exc)
        except Exception:
            LOG.exception("Unexpected bridge error")

        time.sleep(period)


if __name__ == "__main__":
    main()
