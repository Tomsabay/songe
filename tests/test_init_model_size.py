"""Init's model choice against this machine's memory (issue #73)."""

import json
import struct
from types import SimpleNamespace

import pytest

from songe import capacity, cli
from songe.log import Store
from songe.models import ModelRef, registry
from songe.platform import Accel, Memory

GIB = 2**30


def weights(folder, billions):
    """A model folder whose safetensors header declares `billions` of bf16
    parameters (headers only; no weight bytes)."""
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    rows = int(billions * 1e9) // 1000
    header = json.dumps(
        {"w": {"dtype": "BF16", "shape": [rows, 1000], "data_offsets": [0, 0]}}
    ).encode()
    (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    return folder


@pytest.fixture
def mac(tmp_path, monkeypatch):
    """A fresh init on an Apple Silicon Mac with `memory` GiB in total, all
    else faked; run(memory, models, agent_model) returns the store."""
    from songe import agents, launchd
    from songe.backends import install
    from songe.models import discovery

    monkeypatch.setattr(capacity, "_CACHE", {"at": 0.0, "value": None})
    monkeypatch.setattr("songe.upstream.lmstudio.find_lms", lambda: "/fixture/lms")
    monkeypatch.setattr(discovery, "loaded_lmstudio", lambda refs, *a: refs)
    monkeypatch.setattr(discovery, "lmstudio_key_path", lambda *a: None)
    monkeypatch.setattr(install, "plan", lambda *a: [])
    monkeypatch.setattr(install, "conversion_plan", lambda *a: [])
    monkeypatch.setattr(agents, "proposed", lambda *a, **kw: [])
    monkeypatch.setattr(agents, "apply", lambda *a: None)
    monkeypatch.setattr(agents, "routing", lambda home=None: {})
    monkeypatch.setattr(launchd, "install", lambda *a, **kw: None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    def run(memory, models, agent_model=""):
        total = memory * GIB
        monkeypatch.setattr(
            "songe.platform.current",
            lambda: SimpleNamespace(
                name="darwin",
                accelerators=lambda: [Accel("metal", "Apple M4", total, total)],
                memory=lambda: Memory(total, total // 4),
            ),
        )
        refs = [
            ModelRef("lmstudio", name, "safetensors", weights(tmp_path / name, b), "qwen3")
            for name, b in models
        ]
        monkeypatch.setattr(discovery, "lmstudio", lambda **kw: refs)
        monkeypatch.setattr(discovery, "lmstudio_downloads", lambda: tmp_path)
        monkeypatch.setattr(cli, "detect_agent_binding", lambda: (agent_model, ""))
        root = tmp_path / "data"
        args = cli.parser().parse_args(
            ["init", "--upstream", "lmstudio", "--yes", "--launchd-target", str(tmp_path / "svc")]
        )
        cli.initialize(args, root)
        return Store(root)

    return run


def test_model_billions_reads_headers(tmp_path):
    ref = ModelRef("lmstudio", "m", "safetensors", weights(tmp_path / "m", 8), "x")
    assert capacity.model_billions(ref) == pytest.approx(8, rel=0.01)
    missing = ModelRef("lmstudio", "m", "gguf", tmp_path / "none.gguf", "x")
    assert capacity.model_billions(missing) is None


def test_init_skips_a_model_too_big_for_this_mac(mac, capsys):
    # A 16 GB Mac trains up to about 3B: the agent's 8B is still served, but
    # auto-dream starts with the 0.8B, and init says why.
    store = mac(16, [("lfm2.5-8b-a1b", 8.3), ("qwen3.5-0.8b", 0.8)], "lfm2.5-8b-a1b")
    out = capsys.readouterr().out
    assert store.get("primary_model") == "qwen3.5-0.8b"
    assert registry.auto_dream(store, registry.find(store, "qwen3.5-0.8b"))
    assert "auto-dream with qwen3.5-0.8b" in out
    assert "skipped lfm2.5-8b-a1b is about 8B, this computer trains up to about 3B" in out
    # Using the big one later does not switch learning on by itself.
    row = registry.observe(cli.Config.load(store.root), store, "lfm2.5-8b-a1b")
    assert row["learning_level"] == "off" and not registry.auto_dream(store, row)


def test_init_leaves_auto_dream_off_when_no_model_fits(mac, capsys):
    store = mac(16, [("lfm2.5-8b-a1b", 8.3)])
    out = capsys.readouterr().out
    row = registry.find(store, "lfm2.5-8b-a1b")
    # Set up for serving and recording; nothing starts measuring on its own.
    assert cli.Config.load(store.root).public_model == "songe/lfm2.5-8b-a1b"
    assert row["learning_level"] == "new"
    assert registry.auto_dream(store, row) is False
    assert "auto-dream off · lfm2.5-8b-a1b is about 8B" in out
    assert "songe models lfm2.5-8b-a1b on" in out


def test_init_keeps_the_agents_model_when_it_fits(mac, capsys):
    # 24 GB: an 8B is within what this Mac trains; nothing changes.
    store = mac(24, [("lfm2.5-8b-a1b", 8.3), ("qwen3.5-0.8b", 0.8)], "lfm2.5-8b-a1b")
    out = capsys.readouterr().out
    assert store.get("primary_model") == "lfm2.5-8b-a1b"
    assert registry.auto_dream(store, registry.find(store, "lfm2.5-8b-a1b"))
    assert "new conversations · starts when idle" in out and "skipped" not in out
