"""Keep process completion receipts typed until a queued wake actually runs.

Adapter acceptance only queues a wake. The foreground may read its result in the
meantime, so admission-time de-duplication alone cannot prevent a redundant turn.
Private event attributes are deliberate: inbound provider metadata cannot forge
receipt identities or suppress a real user message.
"""
from __future__ import annotations

from gateway.config import Platform
from tools.process_registry_notifications import ProcessNotificationBatch

_SILENCE_GUIDANCE = (
    "[Completion delivery instruction: preserve new failures and actionable results. "
    "If these results are already covered by your previous final and nothing new "
    "requires user attention, respond with exactly NO_REPLY. Do not send an "
    "acknowledgment saying the completion notification was already handled.]"
)


def bind_process_completion_event(event, evt: dict, text: str) -> None:
    if evt.get("type") != "completion":
        return
    entries = evt.get("_process_completion_entries") or ((evt, text),)
    event._gateway_process_completion_batch = ProcessNotificationBatch(
        tuple((dict(raw), rendered) for raw, rendered in entries)
    )
    event.text = text + "\n\n" + _SILENCE_GUIDANCE
    event._gateway_process_completion_text = event.text


def is_process_completion_event(event) -> bool:
    # A merged human message must never be discarded or have its reply anchor changed.
    return bool(
        getattr(event, "internal", False)
        and getattr(event, "_gateway_process_completion_batch", None)
        and event.text == getattr(event, "_gateway_process_completion_text", None)
    )


def completion_reply_has_no_reference(event) -> bool:
    source = getattr(event, "source", None)
    return bool(
        is_process_completion_event(event)
        and source is not None
        and source.platform == Platform.DISCORD
        and source.thread_id
    )


def refresh_process_completion_event(event, *, session_key: str = "", previous_result=None) -> bool:
    """Re-render still-unread entries; False means no model wake is needed.

    wait/log receipts always apply. A poll remains read-only and does not consume
    watcher output; only a successful, visible foreground final for the SAME
    session permits eliding an already-observed successful exit from its queued
    follow-up. Failures, interruption, intentional silence, and unknown receipts
    are fail-open. No output-text heuristic or model-generated acknowledgment is
    used to classify a result as handled.
    """
    if not is_process_completion_event(event):
        return True
    from tools.process_registry import process_registry
    from gateway.response_filters import is_intentional_silence_agent_result

    result = previous_result or {}
    final = result.get("final_response") or ""
    finalized = bool(final.strip() and not is_intentional_silence_agent_result(result, final)
                     and not result.get("failed") and not result.get("interrupted")
                     and not result.get("error"))
    kept = []
    for evt, text in event._gateway_process_completion_batch.notifications:
        sid = str(evt.get("session_id") or "")
        process = process_registry.get(sid)
        epoch = evt.get("started_at")
        same_incarnation = epoch is not None and (process is None or process.started_at == epoch)
        if same_incarnation and process_registry.is_completion_consumed(sid):
            continue
        if finalized and process_registry.has_observed_successful_completion(sid, session_key, epoch):
            continue
        kept.append((evt, text))
    if not kept:
        return False
    if len(kept) == len(event._gateway_process_completion_batch.notifications):
        return True
    event._gateway_process_completion_batch = ProcessNotificationBatch(tuple(kept))
    if len(kept) == 1:
        event.text = kept[0][1]
    else:
        from gateway.run_notifications import GatewayNotificationsMixin
        event.text = GatewayNotificationsMixin._format_coalesced_process_completions(
            [(text, evt, None) for evt, text in kept]
        )
    event.text += "\n\n" + _SILENCE_GUIDANCE
    event._gateway_process_completion_text = event.text
    return True
