"""Completion admission is not execution: recheck receipts before a queued wake."""
import asyncio
import json
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, _reply_anchor_for_event
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_notifications import format_process_notification


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="sent")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def stop_typing(self, chat_id):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def setup_runner(monkeypatch, tmp_path):
    import gateway.run as gw
    import tools.process_registry as pr
    monkeypatch.setattr(gw, "_hermes_home", tmp_path)
    registry = ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    runner = GatewayRunner(GatewayConfig())
    adapter = CaptureAdapter()
    runner.adapters[Platform.DISCORD] = adapter
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread", chat_type="thread",
                           thread_id="thread", message_id="ancient-origin")
    key = build_session_key(source)
    runner.session_store.get_or_create_session(source)
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[key] = asyncio.Event()
    return runner, adapter, registry, source, key


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["direct", "batch", "mixed", "idle", "fifo"])
@pytest.mark.parametrize("receipt,exit_code,previous_result,keep", [
    ("poll", 0, {"final_response": "Task completed"}, False),
    ("wait", 0, {"final_response": "Task completed"}, False),
    ("log", 0, {"final_response": "Task completed"}, False),
    ("none", 0, {"final_response": "Task completed"}, True),
    ("poll", 7, {"final_response": "Task completed"}, True),
    ("poll", 0, {"final_response": "Failed", "failed": True}, True),
    ("poll", 0, {"final_response": "Interrupted", "interrupted": True}, True),
    ("poll", 0, {"final_response": "NO_REPLY"}, True),
    ("foreign-poll", 0, {"final_response": "Task completed"}, True),
    ("merged-user", 0, {"final_response": "Task completed"}, True),
    ("missing-epoch", 0, {"final_response": "Task completed"}, True),
    ("stale-epoch", 0, {"final_response": "Task completed"}, True),
    ("killed", 0, {"final_response": "Task completed"}, True),
    ("unbound-poll", 0, {"final_response": "Task completed"}, True),
    ("poll", 0, {"final_response": "Partial", "partial": True}, True),
    ("poll", 0, {"final_response": "Not complete", "completed": False}, True),
    ("media-log", 0, {"final_response": "Task completed"}, True),
])
async def test_queued_completion_rechecks_inline_result_without_losing_unread_results(
    monkeypatch, tmp_path, receipt, exit_code, previous_result, keep, delivery,
):
    runner, adapter, registry, source, key = setup_runner(monkeypatch, tmp_path)
    # Real native subprocess output, not a fabricated command result.
    completed = subprocess.run([sys.executable, "-c",
        f"import sys; print('completion-result'); sys.exit({exit_code})"],
        capture_output=True, text=True, check=False, timeout=10)
    session = ProcessSession(id="proc-live", command="native completion probe", started_at=time.time(),
        session_key=key, exited=True, exit_code=completed.returncode,
        output_buffer=completed.stdout, notify_on_complete=True)
    registry._finished[session.id] = session
    evt = runner._build_process_completion_event({"session_key": key, "message_id": "old-trigger"},
                                                  session, session.id)
    if receipt == "missing-epoch":
        evt.pop("started_at")
    if receipt == "stale-epoch":
        evt["started_at"] -= 1
    synth_text = format_process_notification(evt)
    assert synth_text is not None
    if delivery in {"batch", "mixed"}:
        runner._completion_notification_batch_window = 0
        tasks = [runner._enqueue_process_completion_notification(synth_text, evt)]
        if delivery == "mixed":
            sibling = ProcessSession(id="proc-failure", command="failure", session_key=key,
                started_at=time.time(), exited=True, exit_code=9, output_buffer="UNREAD-FAILURE")
            registry._finished[sibling.id] = sibling
            sibling_evt = runner._build_process_completion_event(
                {"session_key": key, "message_id": "old-trigger"}, sibling, sibling.id)
            sibling_text = format_process_notification(sibling_evt)
            assert sibling_text is not None
            tasks.append(runner._enqueue_process_completion_notification(sibling_text, sibling_evt))
        assert all(await asyncio.gather(*tasks))
    else:
        assert await runner._inject_watch_notification(synth_text, evt)
    pending = adapter._pending_messages[key]
    # The race: admission happens BEFORE the foreground reads the exit.
    if receipt in {"poll", "foreign-poll", "merged-user", "missing-epoch", "stale-epoch", "killed"}:
        from tools.approval_context import set_current_session_key, reset_current_session_key
        from tools.process_registry import _handle_process
        caller = "agent:main:discord:thread:other:other" if receipt == "foreign-poll" else key
        token = set_current_session_key(caller)
        try:
            # Exercise the real tool boundary; model args cannot forge caller provenance.
            result = json.loads(_handle_process({"action": "poll", "session_id": session.id,
                                                 "observer_session_key": key}))
            assert result["status"] == "exited"
        finally:
            reset_current_session_key(token)
        assert not registry.is_completion_consumed(session.id)
    if receipt == "unbound-poll":
        # A status-only observer and a stale ambient env must not count as the caller.
        monkeypatch.setenv("HERMES_SESSION_KEY", key)
        from tools.approval_context import set_current_session_key, reset_current_session_key
        from tools.process_registry import _handle_process
        token = set_current_session_key("")
        try:
            _handle_process({"action": "poll", "session_id": session.id})
        finally:
            reset_current_session_key(token)
        assert not registry.has_observed_successful_completion(session.id, key, session.started_at)
    if receipt == "killed":
        session.completion_reason = "killed"
    if receipt == "merged-user":
        pending.text += "\nA real new user request must survive"
    if receipt == "wait":
        assert registry.wait(session.id, timeout=1)["status"] == "exited"
    if receipt == "media-log":
        from gateway.platforms.base import merge_pending_message_event
        human_photo = MessageEvent(text="", source=source, message_type=MessageType.PHOTO,
            media_urls=["human-image.png"], media_types=["image/png"], message_id="human-photo")
        merge_pending_message_event(adapter._pending_messages, key, human_photo)
        pending = adapter._pending_messages[key]
        assert pending.media_urls == ["human-image.png"]
        runner._enrich_inbound_images = AsyncMock(side_effect=lambda _source, _key, text, _paths: text)
    if receipt in {"log", "media-log"}:
        registry.read_log(session.id)
    expected_drop = not keep
    if delivery == "idle":
        text = await runner._prepare_profile_scoped_inbound_message_text(
            event=pending, source=source, history=[], session_key=key)
        event = pending if text is not None else None
        keep = receipt not in {"wait", "log"}
    elif delivery == "fifo":
        human = MessageEvent(text="REAL-USER-NEXT", message_type=MessageType.TEXT,
                             source=source, message_id="new-user")
        runner._enqueue_fifo(key, human, adapter)
        event, text = await runner._run_agent_drain_pending(previous_result, adapter, source, key)
        if expected_drop:
            assert event is human and text == "REAL-USER-NEXT"
            assert runner._queue_depth(key, adapter=adapter) == 0
            return
        assert adapter._pending_messages[key] is human
    else:
        event, text = await runner._run_agent_drain_pending(previous_result, adapter, source, key)
    if delivery == "mixed":
        keep = True
        assert text is not None and "UNREAD-FAILURE" in text
        if expected_drop:
            assert "completion-result" not in text
    assert (event is not None) == keep
    assert bool(text) == keep
    if receipt == "media-log":
        assert event is not None and event.media_urls == ["human-image.png"]
    if keep:
        assert text is not None
        if not (delivery == "mixed" and expected_drop):
            assert "completion-result" in text
    else:
        # Nothing reaches the new-model-turn path; no generic empty-response nudge can fire.
        assert text is None


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("outer_kind", ["human", "completion"])
@pytest.mark.parametrize("next_turn", ["completion", "late-log", "nested-human", "nested-completion"])
async def test_discord_completion_final_does_not_quote_original_task(monkeypatch, tmp_path, queued, next_turn, outer_kind):
    runner, adapter, registry, source, key = setup_runner(monkeypatch, tmp_path)
    session = ProcessSession(id="proc-unread", command="probe", session_key=key, started_at=time.time(),
                             exited=True, exit_code=7, output_buffer="new failure")
    registry._finished[session.id] = session
    evt = runner._build_process_completion_event({"session_key": key, "message_id": "old-trigger"},
                                                  session, session.id)
    synth_text = format_process_notification(evt)
    assert synth_text is not None
    assert await runner._inject_watch_notification(synth_text, evt)
    wake = adapter._pending_messages.pop(key)
    assert "exactly NO_REPLY" in wake.text
    assert _reply_anchor_for_event(wake) is None
    assert wake.source.thread_id == "thread"
    outer = MessageEvent(text="Original task", message_type=MessageType.TEXT,
                         source=source, message_id="original-user")
    assert _reply_anchor_for_event(outer) == "original-user"
    result = {"final_response": "New failure", "messages": []}
    if queued:
        runner._run_agent_deliver_first_response = AsyncMock()
        if next_turn == "late-log":
            async def deliver_then_consume(*args):
                await runner._deliver_queued_first_response(
                    "Original final", source, adapter, metadata={"thread_id": "thread"}, deliver_media=False)
                registry.read_log(session.id)
            runner._run_agent_deliver_first_response.side_effect = deliver_then_consume
        elif next_turn.startswith("nested-"):
            result["_suppress_discord_reply_reference"] = next_turn == "nested-completion"
        runner._refresh_agent_cache_message_count = AsyncMock()
        runner._run_agent = AsyncMock(return_value=result)
        ctx = SimpleNamespace(source=source, session_id="sid", session_key=key, run_generation=1,
            _interrupt_depth=0, history=[], _status_thread_metadata={"thread_id": "thread"},
            context_prompt="", result_holder=[None])
        result = await runner._run_agent_queued_followup(ctx, adapter, wake.text, wake,
            {"final_response": "Original final"}, {"final_response": "Original final", "messages": []}, None)
        if next_turn == "late-log":
            runner._run_agent.assert_not_called()
            assert result["final_response"] == ""
            response, silent, messages = await runner._hmwa_shape_agent_response(
                dict(result, api_calls=1), source, [], runner.session_store.get_or_create_session(source),
                key, key, 1, "sid", "discord", time.time())
            assert silent and not response
            delivery = await runner._hmwa_deliver_turn_response(
                outer, source, SimpleNamespace(session_id="sid"), key, 1, result, messages, response, "", silent)
            adapter._active_sessions.pop(key, None)
            adapter.set_message_handler(AsyncMock(return_value=delivery))
            await adapter._process_message_background(outer, key)
            assert adapter.sent == [("thread", "Original final", None, {"thread_id": "thread"})]
            return
    delivery_event = outer if queued else wake
    if queued and outer_kind == "completion":
        delivery_event = wake
    if queued and next_turn == "nested-human":
        delivery_event._gateway_suppress_reply_reference = True
    runner._should_send_voice_reply = lambda *a, **k: False
    response = await runner._hmwa_deliver_turn_response(delivery_event, source,
        SimpleNamespace(session_id="sid"), key, 1, result, [], "New failure", "", False)
    adapter._active_sessions.pop(key, None)
    adapter.set_message_handler(AsyncMock(return_value=response))
    await adapter._process_message_background(delivery_event, key)
    expected_anchor = delivery_event.message_id if queued and next_turn == "nested-human" else None
    assert adapter.sent == [("thread", "New failure", expected_anchor, {"thread_id": "thread", "notify": True})]
