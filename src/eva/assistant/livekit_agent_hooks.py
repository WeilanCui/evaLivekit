"""Reusable agent-side hooks for evaluating a LiveKit agent with EVA.

Drop-in glue for **any** LiveKit Agents app that wants to be evaluated by EVA's
``framework=livekit`` server (``eva.assistant.livekit_server``). The bridge joins
your agent's room and records the conversation, but two things only your agent
can see — the tool calls it executes and its LLM token usage — must be forwarded
into the room. This module does that in one call:

    from eva.assistant.livekit_agent_hooks import attach_eva_telemetry

    session = AgentSession(...)
    attach_eva_telemetry(session, room, ctx=ctx, model="gpt-4o")
    await session.start(...)

``attach_eva_telemetry`` is a no-op unless the job was dispatched by the EVA
bridge (it tags dispatch/room metadata with ``{"source": "eva"}``), so it is safe
to call unconditionally in production.

Design notes:
- **Dependency-free**: stdlib only, and ``session``/``room`` are duck-typed, so
  this module imports nothing from ``eva`` or ``livekit`` and can be imported
  into any agent runtime without pulling in EVA's (heavy) dependencies.
- **Best-effort**: every hook swallows its own errors — telemetry forwarding
  must never disrupt the live call.
- The topics + payload shapes here are the contract consumed by
  ``LiveKitAssistantServer`` (``_TOOL_CALLS_TOPIC`` / ``_METRICS_TOPIC``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger("eva.livekit_agent_hooks")

# Text-stream topics the EVA bridge (LiveKitAssistantServer) listens on.
TOOL_CALLS_TOPIC = "lk.tool_calls"
METRICS_TOPIC = "lk.metrics"

# Dispatch/room metadata key the bridge sets to mark an eval run.
_EVA_SOURCE = "eva"


def is_eva_run(ctx: Any) -> bool:
    """True when this job was dispatched by the EVA eval bridge.

    The bridge tags the dispatch (hence job) metadata — and, defensively, room
    metadata — with ``{"source": "eva", ...}``. ``ctx`` is the agent
    ``JobContext`` (duck-typed: we only read ``.job.metadata`` / ``.room.metadata``).
    """
    candidates = [
        getattr(getattr(ctx, "job", None), "metadata", None),
        getattr(getattr(ctx, "room", None), "metadata", None),
    ]
    for meta in candidates:
        if not meta:
            continue
        try:
            if json.loads(meta).get("source") == _EVA_SOURCE:
                return True
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return False


def attach_eva_telemetry(
    session: Any,
    room: Any,
    *,
    ctx: Any = None,
    model: str = "",
    force: bool = False,
) -> bool:
    """Wire tool-call + token-usage forwarding from a LiveKit agent to EVA.

    Args:
        session: the ``AgentSession`` (duck-typed; uses ``.on(event, cb)``).
        room: the LiveKit ``Room`` (uses ``.local_participant.send_text``).
        ctx: the agent ``JobContext`` used to gate on :func:`is_eva_run`. If
            omitted, gating is skipped (caller is responsible for gating).
        model: model identifier recorded alongside token usage.
        force: attach even if ``is_eva_run(ctx)`` is False.

    Returns:
        True if the hooks were attached, False if skipped (not an eval run).
    """
    if ctx is not None and not force and not is_eva_run(ctx):
        return False

    def _publish(topic: str, payload: dict[str, Any]) -> None:
        text = json.dumps(payload)

        async def _send() -> None:
            try:
                await room.local_participant.send_text(text, topic=topic)
            except Exception:
                logger.exception("EVA telemetry: failed to publish on %s", topic)

        try:
            asyncio.create_task(_send())
        except RuntimeError:
            # No running loop (shouldn't happen inside a session callback).
            logger.debug("EVA telemetry: no running loop to publish on %s", topic)

    def _on_tools_executed(ev: Any) -> None:
        try:
            for call, output in ev.zipped():
                _publish(
                    TOOL_CALLS_TOPIC,
                    {
                        "name": getattr(call, "name", ""),
                        "arguments": getattr(call, "arguments", "") or "",
                        "result": getattr(output, "output", "") or "",
                        "is_error": getattr(output, "is_error", False),
                        "call_id": getattr(call, "call_id", ""),
                    },
                )
        except Exception:
            logger.exception("EVA telemetry: error handling function_tools_executed")

    def _on_metrics(ev: Any) -> None:
        try:
            metrics = getattr(ev, "metrics", None)
            if getattr(metrics, "type", "") != "llm_metrics":
                return
            _publish(
                METRICS_TOPIC,
                {
                    "model": model,
                    "prompt_tokens": getattr(metrics, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(metrics, "completion_tokens", 0) or 0,
                },
            )
        except Exception:
            logger.exception("EVA telemetry: error handling metrics_collected")

    session.on("function_tools_executed", _on_tools_executed)
    session.on("metrics_collected", _on_metrics)
    logger.info("EVA telemetry hooks attached (tool calls + token usage)")
    return True
