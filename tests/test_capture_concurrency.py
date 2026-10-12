"""Overlapping captures must not race OCR, clipboard copying or history."""

from __future__ import annotations

import importlib
import json
import queue
import sys
import threading
import types
import urllib.error
import urllib.request

import pytest
from PIL import Image

from snipmd import backends, history, system
from snipmd.config import Config
from snipmd.engine import Engine
from snipmd.server import make_server


class Probe:
    def __init__(self, stage):
        self.stage = stage
        self.entered = threading.Event()
        self.release = threading.Event()
        self.captures = 0
        self.ocr = 0
        self.copies = []
        self.records = 0

    def pause(self, stage, count):
        if stage == self.stage and count == 1:
            self.entered.set()
            assert self.release.wait(5), f"{stage} release timed out"

    def capture(self, dest):
        self.captures += 1
        count = self.captures
        Image.new("RGB", (120, 60), (40 * count, 0, 0)).save(dest)
        self.pause("capture", count)
        return dest

    def load(self):
        pass

    name = "probe"

    def recognize(self, path, prompt, max_tokens):
        self.ocr += 1
        count = self.ocr
        with Image.open(path) as image:
            color = image.getpixel((image.width // 2, image.height // 2))[0]
        self.pause("ocr", count)
        return f"capture-{color}"

    def copy(self, text):
        self.copies.append(text)
        self.pause("clipboard", len(self.copies))
        return True


@pytest.fixture
def setup_probe(monkeypatch, tmp_path):
    monkeypatch.setenv("SNIPMD_HOME", str(tmp_path / "home"))

    def make(stage):
        probe = Probe(stage)
        monkeypatch.setattr(system, "capture_region", probe.capture)
        monkeypatch.setattr(system, "copy_to_clipboard", probe.copy)
        monkeypatch.setattr(backends, "pick", lambda cfg, name=None: probe)
        real_add = history.add

        def add(**kwargs):
            probe.records += 1
            probe.pause("history", probe.records)
            return real_add(**kwargs)

        monkeypatch.setattr(history, "add", add)
        return probe

    return make


@pytest.fixture
def app_module(monkeypatch):
    # macOS UI is unavailable on Linux. Exercise the shipped worker methods.
    monkeypatch.setitem(sys.modules, "rumps", types.SimpleNamespace(App=object))
    sys.modules.pop("snipmd.app", None)
    module = importlib.import_module("snipmd.app")
    yield module
    sys.modules.pop("snipmd.app", None)


@pytest.mark.parametrize("stage", ["capture", "ocr", "clipboard", "history"])
def test_hotkey_rejects_overlap_through_completion(stage, setup_probe, app_module, monkeypatch):
    probe = setup_probe(stage)
    instance = app_module.SnipApp.__new__(app_module.SnipApp)
    instance.cfg = Config()
    instance.engine = Engine(instance.cfg)
    instance.events = queue.Queue()
    instance.busy = threading.Lock()
    threads = []

    def spawn(*args, **kwargs):
        thread = threading.Thread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(app_module, "threading", types.SimpleNamespace(Thread=spawn))
    try:
        instance._on_hotkey()
        assert probe.entered.wait(5), f"{stage} was not reached"
        instance._on_hotkey()
        instance.on_snip(None)
        assert len(threads) == 1, "an overlapping hotkey or menu call started a second worker"
        assert probe.captures == 1
        probe.release.set()
        threads[0].join(5)
        assert not threads[0].is_alive()
        assert probe.copies == ["capture-40"]
        assert [entry.text for entry in history.recent()] == ["capture-40"]
        instance._on_hotkey()
        threads[-1].join(5)
        assert not threads[-1].is_alive()
        assert probe.captures == probe.ocr == probe.records == 2
        assert probe.copies == ["capture-40", "capture-80"]
        assert [entry.text for entry in history.recent()] == ["capture-80", "capture-40"]
        assert not instance.busy.locked()
        assert not any(event[0] == "error" for event in list(instance.events.queue))
    finally:
        probe.release.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()


def call(url):
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.loads(exc.read())


@pytest.mark.parametrize("stage", ["capture", "ocr", "history"])
def test_http_snip_rejects_overlap_until_history_recorded(stage, setup_probe):
    probe = setup_probe(stage)
    server = make_server(Engine(Config()), "127.0.0.1", 0)
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/snip"
    answers = []
    pending = threading.Thread(target=lambda: answers.append(call(url)))
    try:
        pending.start()
        assert probe.entered.wait(5)
        assert call(url) == (409, {"error": "a snip is already in progress"})
        assert probe.captures == 1
        probe.release.set()
        pending.join(5)
        assert not pending.is_alive()
        assert answers[0][0] == 200
        assert answers[0][1]["text"] == "capture-40"
        assert call(url)[1]["text"] == "capture-80"
        assert probe.captures == probe.ocr == probe.records == 2
        assert probe.copies == []  # HTTP /snip returns OCR, it does not copy it.
        assert [entry.text for entry in history.recent()] == ["capture-80", "capture-40"]
    finally:
        probe.release.set()
        pending.join(5)
        server.shutdown()
        server.server_close()
        serving.join(5)
        assert not pending.is_alive() and not serving.is_alive()
