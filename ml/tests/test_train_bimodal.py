from __future__ import annotations

import ast
import contextlib
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch
import yaml

import train_bimodal
from data.bimodal_raster import encode_png, render_strokes_to_strip
from data.bimodal_shards import (
    BimodalAugmentConfig, ShardWriter, collate_bimodal_batch, encode_npz, iter_shard, sha256_file,
    teacher_stroke_features,
)
from data.preprocessing import PreprocessingConfig
from model.bimodal import BimodalMathModel, BimodalModelConfig
from model.stroke2latex import ModelConfig, Stroke2Latex
from training.checkpoint import save_checkpoint
from model.stroke_structure import structure_loss
from tokenizer.latex_tokenizer import LatexTokenizer
from tokenizer.normalization import LABEL_VERSION, normalize_training_latex
from tokenizer.standardize import standardize_latex
from tools import export_drive_ids

ML = Path(__file__).resolve().parents[1]
CONFIG = ML / "configs" / "bimodal_g4.yaml"
STAGE25_CONFIG = ML / "configs" / "bimodal_stage25_3.yaml"
STAGE26_CONFIG = ML / "configs" / "bimodal_stage26.yaml"
NOTEBOOK = ML / "colab" / "train_bimodal_g4.ipynb"
DATASETS = Path(r"E:\math-recogniser-datasets")
LABELS = ["x+1", "a^{2}+b^{2}", r"\frac{1}{2}", "y=2x", "z_{1}", "C+c", r"\sqrt{2}", "1,2", "X-x", "u+v"]
SYMBOLS = ["x", "+", "1"]
RELATIONS = ["None", "Right", "Sup"]
COMMON = ["--preset", "bimodal_tiny", "--no-pretrained", "--device", "cpu", "--log-every", "1",
          "--max-wall-minutes", "0"]
REAL_TOKEN = "hf_" + "A1b2C3d4" * 4
HF_REPO = "me/math-recogniser-runs"


def glyph_strokes(seed: int, glyphs: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    strokes = []
    for glyph in range(glyphs):
        for _ in range(int(rng.integers(1, 3))):
            t = np.linspace(0, 1, int(rng.integers(8, 16)))
            phase = rng.uniform(0, 6.28)
            x = glyph * 120 + 20 + 60 * t + 15 * np.sin(6.28 * t + phase)
            y = 100 + 60 * np.cos(3.14 * t + phase)
            strokes.append(np.stack([x, y], axis=1))
    return strokes


def stroke_features(strokes: list[np.ndarray]) -> np.ndarray:
    points = np.concatenate(strokes)
    origin, scale = points.min(0), max(float(np.ptp(points[:, 1])), 1.0)
    rows, previous = [], None
    for index, stroke in enumerate(strokes):
        for position, (x, y) in enumerate((stroke - origin) / scale):
            dx, dy = (0.0, 0.0) if previous is None else (x - previous[0], y - previous[1])
            rows.append([x, y, dx, dy, 0.01, 0.0, 0.0, position == 0, position == len(stroke) - 1,
                         index / max(1, len(strokes) - 1)])
            previous = (x, y)
    return np.asarray(rows, dtype=np.float16)


def make_sample(seed: int, source: str, split: str, *, ink: bool, structure: bool = False,
                hint: float | None = None) -> tuple[dict, bytes, bytes | None]:
    label = LABELS[seed % len(LABELS)]
    strokes = glyph_strokes(seed)
    image, width = render_strokes_to_strip(strokes)
    synthetic = source.endswith("synthetic")
    meta = {"id": f"{source}:{seed}", "source": source, "split": split, "latex": label,
            "normalized_latex": normalize_training_latex(label), "has_strokes": ink, "has_timing": ink,
            "image_width": width, "stroke_count": len(strokes), "writer_id": f"w{seed % 3}",
            "synthetic": synthetic, "rendered": True}
    if hint:
        meta["upsample_hint"] = hint
    npz = None
    if ink:
        arrays = {"features": stroke_features(strokes)}
        if structure:
            groups = np.asarray([min(index, 2) for index in range(len(strokes))], dtype=np.int16)
            arrays.update(stroke_group_ids=groups, stroke_symbol_ids=np.asarray([0, 1, 2], dtype=np.int16)[groups],
                          relation_edges=np.asarray([[0, 1, 1], [1, 2, 2]], dtype=np.int16))
        npz = encode_npz(**arrays)
    return meta, encode_png(image), npz


def write_stage(root: Path, directory: str, samples: list[tuple[dict, bytes, bytes | None]]) -> dict:
    shards, sources = [], {}
    for start in range(0, len(samples), 8):
        writer = ShardWriter(root / directory / f"shard-{start // 8:05d}.tar")
        for offset, (meta, png, npz) in enumerate(samples[start:start + 8]):
            writer.write(f"{start + offset:08d}", meta, png, npz)
            sources[meta["source"]] = sources.get(meta["source"], 0) + 1
        shards.append(writer.close())
    return {"directory": directory, "samples": len(samples), "sources": sources, "shards": shards,
            "with_structure": sum(b"stroke_group_ids" in (npz or b"") for _, _, npz in samples)}


def build_shards(root: Path) -> Path:
    seed = iter(range(10_000))
    stage1 = [make_sample(next(seed), "crohme_train", "train", ink=False, hint=2.0) for _ in range(8)]
    stage1 += [make_sample(next(seed), "mathwriting_train", "train", ink=False) for _ in range(8)]
    stage1 += [make_sample(next(seed), "hme100k", "train", ink=False) for _ in range(8)]
    stage2 = [make_sample(next(seed), "crohme_train", "train", ink=True, structure=index < 4) for index in range(8)]
    stage2 += [make_sample(next(seed), "mathwriting_train", "train", ink=True) for _ in range(6)]
    stage2 += [make_sample(next(seed), "mathwriting_synthetic", "synthetic", ink=True) for _ in range(6)]
    stage2 += [make_sample(next(seed), "hme100k", "train", ink=False) for _ in range(4)]
    val = [make_sample(next(seed), "crohme_val", "validation", ink=True) for _ in range(4)]
    val += [make_sample(next(seed), "mathwriting_val", "validation", ink=True) for _ in range(4)]
    rng = np.random.default_rng(0)
    for samples in (stage1, stage2):
        order = rng.permutation(len(samples))
        samples[:] = [samples[index] for index in order]
    stages = {"stage1": write_stage(root, "stage1_train", stage1), "stage2": write_stage(root, "stage2_train", stage2),
              "val": write_stage(root, "val_dev", val)}
    tokenizer = LatexTokenizer.build(meta["normalized_latex"] for meta, _, _ in stage1 + stage2 + val)
    tokenizer.label_normalization = LABEL_VERSION
    tokenizer.save(root / "tokenizer.json")
    manifest = {"format_version": "bimodal-shards-v1", "smoke": True, "stages": stages,
                "structure": {"symbols": SYMBOLS, "relation_classes": RELATIONS, "records": 4},
                "tokenizer": {"file": "tokenizer.json", "size": len(tokenizer)}, "args": {"crohme_upsample": 2.0}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def write_config(path: Path) -> Path:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tiny = {"batch_size": 4, "num_workers": 0, "amp_dtype": "float32"}
    config["stage1"].update(tiny, epochs=2, lr=1e-3, vision_lr=1e-3)
    config["stage2"].update(tiny, epochs=3, cooldown_epochs=1, lr=1e-3, vision_lr=5e-4, cooldown_lr=3e-5,
                            vision_drop_prob=0.3, stroke_drop_prob=0.3)
    config["stage3"].update(tiny, epochs=1, batch_size=2, grad_accum=2, group_size=4, max_gen_tokens=12,
                            lr=1e-3, temperature=1.0)
    config["validation"].update(batch_size=8, max_gen_tokens=12)
    config["runtime"].update(sync_every_minutes=0, keep_last_checkpoints=1)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run(*args) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = train_bimodal.main([str(arg) for arg in args])
    assert code == 0
    return output.getvalue()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FakeHub:
    """In-memory stand-in for one HF dataset repo behind ``HfApi`` and ``hf_hub_download``."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.commits: list[tuple[str, list[str]]] = []
        self.repos: list[dict] = []
        self.tokens: list[str | None] = []
        self.downloads: list[str] = []
        self.fail_commits = self.squashes = 0

    def api(self, token=None):
        self.tokens.append(token)
        hub = self

        class Api:
            def create_repo(self, **kwargs):
                hub.repos.append(kwargs)

            def create_commit(self, repo_id, operations, *, commit_message, repo_type=None, **_):
                assert repo_type == "dataset"
                if hub.fail_commits:
                    hub.fail_commits -= 1
                    raise ConnectionError("network unreachable")
                for operation in operations:
                    hub.files[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
                hub.commits.append((commit_message, sorted(operation.path_in_repo for operation in operations)))

            def list_repo_files(self, repo_id, *, repo_type=None, **_):
                assert repo_type == "dataset"
                return sorted(hub.files)

            def super_squash_history(self, **kwargs):
                assert kwargs["repo_type"] == "dataset"
                hub.squashes += 1

        return Api()

    def download(self, repo_id, filename, *, repo_type=None, token=None, local_dir=None, **_):
        assert repo_type == "dataset" and token == REAL_TOKEN and local_dir is not None
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        self.downloads.append(filename)
        return str(path)


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", fake.api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.download)
    monkeypatch.setattr(train_bimodal.HFSync, "RETRY_DELAYS", (0.0, 0.0, 0.0))
    monkeypatch.setenv("HF_TOKEN", REAL_TOKEN)
    return fake


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("bimodal_train")
    return SimpleNamespace(root=root, shards=build_shards(root / "shards"), config=write_config(root / "config.yaml"))


@pytest.fixture(scope="module")
def stage1(workspace):
    output, drive = workspace.root / "stage1", workspace.root / "drive" / "stage1"
    logs = run("--config", workspace.config, "--stage", 1, "--shards", workspace.shards, "--output", output,
               "--drive-sync-dir", drive, *COMMON)
    return SimpleNamespace(output=output, drive=drive, logs=logs)


@pytest.fixture(scope="module")
def stage2(workspace, stage1):
    output = workspace.root / "stage2"
    logs = run("--config", workspace.config, "--stage", 2, "--shards", workspace.shards, "--output", output,
               "--init-from", stage1.output / "best.pt", *COMMON)
    return SimpleNamespace(output=output, logs=logs)


@pytest.fixture(scope="module")
def stage3(workspace, stage2):
    mine_logs = run("--config", workspace.config, "--mine-hard", "--shards", workspace.shards,
                    "--init-from", stage2.output / "best.pt", "--mine-limit", 6, "--anchor-limit", 4,
                    "--device", "cpu", "--batch-size", 8)
    output = workspace.root / "stage3"
    logs = run("--config", workspace.config, "--stage", 3, "--shards", workspace.shards, "--output", output,
               "--init-from", stage2.output / "best.pt", *COMMON)
    return SimpleNamespace(output=output, logs=logs, mine_logs=mine_logs)


# --------------------------------------------------------------------------- recipe and units


def test_g4_recipe_matches_budgeted_plan():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == 1337
    assert config["model"] == {"preset": "bimodal_75m", "position_encoding": "relative", "coverage": True,
                               "r2l_decoder_layers": 3, "max_points": 2048, "max_tokens": 512,
                               "vision_backbone": "convnext_tiny.fb_in22k_ft_in1k_384", "vision_pretrained": True}
    one, two, three = config["stage1"], config["stage2"], config["stage3"]
    assert (one["epochs"], one["batch_size"], one["lr"], one["vision_lr"], one["modality_mode"]) == \
        (10, 192, 3.0e-4, 1.0e-4, "vision")
    assert (two["epochs"], two["cooldown_epochs"], two["cooldown_lr"], two["batch_size"]) == (18, 2, 3.0e-5, 128)
    assert two["structure_loss_weights"] == {"symbol": 0.2, "group": 0.1, "relation": 0.1}
    assert three["batch_size"] * three["group_size"] == 128 and three["grad_accum"] == 2
    assert three["reward_weights"] == {"exact_norm": 1.0, "strict_case_bonus": 0.25, "structure_skeleton": 0.5,
                                       "edit_penalty": 0.5}
    assert config["validation"]["primary_metric"] == "normalized_exact"
    assert config["runtime"] == {"sync_every_minutes": 20, "default_max_wall_minutes": 180,
                                 "keep_last_checkpoints": 2}


def test_stage25_3_recipe_matches_local_plan():
    config = yaml.safe_load(STAGE25_CONFIG.read_text(encoding="utf-8"))
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == 1337 and config["standardize_labels"] is True
    assert config["model"] == {**base["model"], "vision_pretrained": False}
    two, three = config["stage2"], config["stage3"]
    assert (two["epochs"], two["cooldown_epochs"], two["cooldown_lr"], two["cooldown_vision_drop_prob"]) == \
        (3, 1, 1.0e-5, 0.15)
    assert two["batch_size"] * two["grad_accum"] == 64
    assert two["allowed_sources"] == ["crohme_train", "mathwriting_train"]
    assert two["source_upsample"] == {"crohme_train": 8.0, "mathwriting_train": 1.0}
    assert (two["lr"], two["stroke_lr"], two["vision_lr"], two["min_lr_ratio"]) == (2.5e-5, 8.0e-5, 5.0e-6, 0.1)
    assert (two["vision_drop_prob"], two["stroke_drop_prob"]) == (0.5, 0.05)
    assert two["structure_loss_weights"] == {"symbol": 0.25, "group": 0.15, "relation": 0.15}
    assert three["batch_size"] * three["grad_accum"] == 32 and three["group_size"] == 8
    assert three["freeze_vision"] is True and three["vision_lr"] == 0.0 and three["min_lr_ratio"] == 0.1
    assert (three["kl_beta"], three["sft_anchor_weight"], three["r2l_anchor_weight"]) == (0.15, 0.15, 0.05)
    assert three["reward_weights"] == train_bimodal.DEFAULT_REWARD_WEIGHTS
    validation = config["validation"]
    assert validation["eval_every_steps"] == 1000 and validation["log_modalities"] is True
    assert config["runtime"] == {"sync_every_minutes": 30, "default_max_wall_minutes": 0,
                                 "keep_last_checkpoints": 3}


def test_stage26_recipe_matches_local_plan():
    config = yaml.safe_load(STAGE26_CONFIG.read_text(encoding="utf-8"))
    stage25 = yaml.safe_load(STAGE25_CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == 1337 and config["standardize_labels"] is True
    assert config["model"] == stage25["model"] and set(config) == {"seed", "standardize_labels", "model", "stage2",
                                                                    "validation", "runtime"}
    two = config["stage2"]
    assert two["standardize_labels"] is True
    assert (two["epochs"], two["cooldown_epochs"], two["cooldown_lr"], two["cooldown_vision_drop_prob"]) == \
        (5, 1, 8.0e-6, 0.15)
    assert two["cooldown_disable_band_mask"] is True and two["cooldown_teacher_kd_weight"] == 0.15
    assert (two["batch_size"], two["grad_accum"], two["shuffle_buffer"]) == (8, 8, 4096)
    assert two["allowed_sources"] == ["crohme_train", "mathwriting_train"]
    assert two["source_upsample"] == {"crohme_train": 8.0, "mathwriting_train": 1.0}
    assert (two["lr"], two["stroke_lr"], two["vision_lr"], two["weight_decay"], two["warmup_ratio"],
            two["min_lr_ratio"]) == (2.0e-5, 6.0e-5, 4.0e-6, 0.01, 0.05, 0.1)
    assert (two["label_smoothing"], two["r2l_weight"], two["modality_mode"], two["vision_drop_prob"],
            two["stroke_drop_prob"], two["ema_decay"]) == (0.02, 0.3, "auto", 0.45, 0.05, 0.999)
    assert (two["max_tokens"], two["grad_clip"], two["num_workers"], two["amp_dtype"]) == (256, 1.0, 4, "bfloat16")
    assert two["structure_loss_weights"] == {"symbol": 0.25, "group": 0.15, "relation": 0.15}
    assert two["teacher_checkpoint"] == \
        "E:/math-recogniser-training/checkpoints/crohme_teacher_20260926_pretrain/best.pt"
    assert (two["teacher_kd_weight"], two["teacher_kd_temp"], two["teacher_kd_min_target_prob"]) == (0.35, 2.0, 0.05)
    augment = BimodalAugmentConfig.from_dict(two["augment"])
    assert augment.coupled_geometry and (augment.rotation_degrees, augment.shear) == (4.5, 0.08)
    assert augment.anisotropic_scale == (0.88, 1.12) and augment.vision_band_mask_width == (16, 40)
    assert (augment.stroke_jitter, augment.point_dropout, augment.vision_band_mask_prob,
            augment.vision_band_mask_max_fraction) == (0.002, 0.015, 0.5, 0.25)
    assert (augment.timing_dropout, augment.morphology_probability, augment.intensity_jitter) == (0.5, 0.35, 0.12)
    assert config["validation"] == {"batch_size": 64, "max_gen_tokens": 256, "eval_every_epochs": 1,
                                    "eval_every_steps": 1000, "log_modalities": True,
                                    "primary_metric": "normalized_exact"}
    trainer = SimpleNamespace(cfg=two)
    assert train_bimodal.Trainer.kd_weight(trainer, False) == 0.35
    assert train_bimodal.Trainer.kd_weight(trainer, True) == 0.15
    assert train_bimodal.Trainer.augment_config(trainer, False).vision_band_mask_prob == 0.5
    cooled = train_bimodal.Trainer.augment_config(trainer, True)
    assert cooled.vision_band_mask_prob == 0.0 and cooled.coupled_geometry
    milder = SimpleNamespace(cfg={**two, "cooldown_augment": {"rotation_degrees": 2.0}, "cooldown_teacher_kd_weight": None})
    assert train_bimodal.Trainer.augment_config(milder, True).rotation_degrees == 2.0
    assert train_bimodal.Trainer.kd_weight(milder, True) == 0.35
    assert train_bimodal.Trainer.augment_config(SimpleNamespace(cfg={}), True) is None


STUDENT_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "x", "+", "1", r"\foo", "y"]
TEACHER_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "y", r"\bar", "1", "x", "+"]


class FixedTeacher(torch.nn.Module):
    """Returns preset teacher-vocabulary logits and records what it was fed."""

    def __init__(self, logits: torch.Tensor, max_points: int, max_tokens: int):
        super().__init__()
        self.config = SimpleNamespace(max_points=max_points, max_tokens=max_tokens)
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.logits, self.calls = logits, []

    def forward(self, features, padding, tokens, token_padding):
        self.calls.append({"features": features, "padding": padding, "tokens": tokens,
                           "token_padding": token_padding})
        return self.logits + self.weight


def kd_batch(student: LatexTokenizer) -> dict:
    ids = student.token_to_id
    rows = [["x", "+", "1"], ["y", r"\foo", "x"], ["1"], ["x"]]
    sequences = [torch.tensor([student.bos_id, *[ids[token] for token in row], student.eos_id]) for row in rows]
    tokens = torch.full((4, 5), student.pad_id)
    for index, sequence in enumerate(sequences):
        tokens[index, :len(sequence)] = sequence
    points = [4, 3, 20, 0]
    features = torch.zeros(4, 20, 10)
    padding = torch.ones(4, 20, dtype=torch.bool)
    for index, count in enumerate(points):
        features[index, :count, :2] = torch.randn(count, 2)
        padding[index, :count] = False
    return {"tokens": tokens, "token_padding_mask": tokens.eq(student.pad_id),
            "has_strokes": torch.tensor([True, True, True, False]),
            "teacher_features": features, "teacher_padding_mask": padding}


def test_teacher_kd_maps_shared_vocabulary_and_gates_on_agreement():
    student, teacher = (LatexTokenizer.from_dict({"tokens": tokens}) for tokens in (STUDENT_TOKENS, TEACHER_TOKENS))
    t = teacher.token_to_id
    favoured = [["x", "+", "y", "<EOS>"], ["y", "x", "x", "<EOS>"]]
    logits = torch.full((2, 4, len(teacher)), -4.0)
    for row, names in enumerate(favoured):
        for position, name in enumerate(names):
            logits[row, position, t[name]] = 4.0
    fixed = FixedTeacher(logits, max_points=16, max_tokens=8)
    kd = train_bimodal.TeacherKD(fixed, teacher, student, PreprocessingConfig(), torch.device("cpu"), "float32")
    assert kd.student_to_teacher.tolist() == [0, 1, 2, 3, 7, 8, 6, -1, 4]
    assert kd.teacher_input.tolist() == [0, 1, 2, 3, 7, 8, 6, teacher.unk_id, 4]
    assert kd.shared_student_ids.tolist() == [2, 4, 5, 6, 8] and kd.shared_teacher_ids.tolist() == [2, 7, 8, 6, 4]
    assert kd.student_to_shared.tolist() == [-1, -1, 0, -1, 1, 2, 3, -1, 4]
    assert kd.summary()["shared_tokens"] == 5 and not fixed.weight.requires_grad

    batch = kd_batch(student)
    student_logits = torch.randn(4, 4, len(student), requires_grad=True)
    loss, stats = kd(batch, student_logits, temperature=2.0, min_target_prob=0.05)
    call = fixed.calls[-1]
    assert call["features"].shape == (2, 4, 10) and call["tokens"].shape == (2, 4)
    assert call["tokens"][1].tolist() == [teacher.bos_id, t["y"], teacher.unk_id, t["x"]]
    assert stats["kd_rows"] == 2 and stats["kd_coverage"] == pytest.approx(6 / 8)
    assert loss > 0 and stats["kd_loss"] == pytest.approx(float(loss.detach()))
    loss.backward()
    grad = student_logits.grad
    assert fixed.weight.grad is None
    assert not grad[2:].any() and not grad[:, :, [0, 1, 3, 7]].any()
    assert not grad[0, 2].any() and not grad[1, 1].any()
    assert grad[0, 0].any() and grad[1, 0].any() and grad[1, 3].any()

    matched = torch.full((4, 4, len(student)), -20.0)
    matched[:2, :, kd.shared_student_ids] = logits[:, :, kd.shared_teacher_ids]
    assert float(kd(batch, matched, 2.0, 0.05)[0]) == pytest.approx(0.0, abs=1e-6)
    assert kd(batch, student_logits, 2.0, 0.0)[1]["kd_coverage"] == pytest.approx(7 / 8)
    blocked, blocked_stats = kd(batch, student_logits, 2.0, 1.01)
    assert float(blocked) == 0.0 and blocked_stats["kd_coverage"] == 0.0

    calls = len(fixed.calls)
    assert kd({**batch, "teacher_features": None}, student_logits, 2.0, 0.05)[1] == {"kd_rows": 0.0}
    nothing = {**batch, "has_strokes": torch.zeros(4, dtype=torch.bool)}
    empty, empty_stats = kd(nothing, student_logits, 2.0, 0.05)
    assert float(empty) == 0.0 and empty_stats == {"kd_rows": 0.0} and len(fixed.calls) == calls


def tiny_teacher(tokenizer: LatexTokenizer, max_points: int = 256) -> Stroke2Latex:
    torch.manual_seed(0)
    return Stroke2Latex(ModelConfig(vocab_size=len(tokenizer), d_model=32, encoder_layers=1, decoder_layers=1,
                                    attention_heads=4, feedforward_dim=64, max_points=max_points, max_tokens=64))


def test_teacher_kd_with_real_stroke2latex_keeps_teacher_frozen():
    student = LatexTokenizer.build(normalize_training_latex(label) for label in LABELS)
    teacher_tokens = LatexTokenizer.build(normalize_training_latex(label) for label in LABELS[:5])
    preprocessing = PreprocessingConfig(max_points=40, min_resampled_points_per_stroke=8)
    kd = train_bimodal.TeacherKD(tiny_teacher(teacher_tokens, max_points=40), teacher_tokens, student, preprocessing,
                                 torch.device("cpu"), "float32")
    samples = []
    for seed in range(3):
        meta, png, npz = make_sample(seed, "crohme_train", "train", ink=True)
        glyph = np.load(io.BytesIO(npz))["features"].astype(np.float32)
        converted = teacher_stroke_features(glyph, preprocessing)
        if seed == 2:
            converted = np.concatenate([converted] * 30)
        tokens = torch.tensor(student.encode(meta["normalized_latex"]))
        samples.append({"image": torch.zeros(2, 128, 64, dtype=torch.uint8), "image_width": 64,
                        "stroke_features": torch.from_numpy(glyph), "teacher_features": torch.from_numpy(converted),
                        "has_timing": True, "tokens": tokens, "r2l_tokens": tokens, "structure_targets": None,
                        "id": str(seed), "source": "crohme_train", "latex": "", "normalized_latex": "",
                        "synthetic": False})
    batch = collate_bimodal_batch(samples, student.pad_id)
    assert batch["teacher_features"].shape[0] == 3 and int((~batch["teacher_padding_mask"][2]).sum()) > 40
    head = torch.nn.Linear(4, len(student))
    student_logits = head(torch.randn(3, batch["tokens"].shape[1] - 1, 4))
    loss, stats = kd(batch, student_logits, 2.0, 0.0)
    assert stats["kd_rows"] == 2 and 0 < stats["kd_coverage"] <= 1 and torch.isfinite(loss) and loss > 0
    loss.backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in kd.model.parameters())
    assert not kd.model.training


def test_parameter_groups_split_stroke_learning_rate():
    model = BimodalMathModel(train_bimodal.build_model_config(
        {"model": {"preset": "bimodal_tiny"}}, LatexTokenizer.build(LABELS), {}, 2, pretrained=False))
    legacy = train_bimodal.parameter_groups(model, 1e-3, 1e-4, 0.01)
    assert {group["name"].split("_")[0] for group in legacy} <= {"main", "vision"}
    groups = train_bimodal.parameter_groups(model, 1e-3, 1e-4, 0.01, stroke_lr=5e-3)
    rates = {group["name"]: group["base_lr"] for group in groups}
    assert rates["main_decay"] == 1e-3 and rates["stroke_decay"] == 5e-3 and rates["vision_decay"] == 1e-4
    assert groups[0]["name"].startswith("main")
    stroke = {id(parameter) for parameter in model.stroke_encoder.parameters()}
    assert stroke == {id(parameter) for group in groups if group["name"].startswith("stroke")
                      for parameter in group["params"]}
    assert sum(len(group["params"]) for group in groups) == sum(len(group["params"]) for group in legacy)
    model.vision_encoder.requires_grad_(False)
    frozen = train_bimodal.parameter_groups(model, 1e-3, 0.0, 0.01, stroke_lr=5e-3)
    assert not any(group["name"].startswith("vision") for group in frozen)


def test_standardized_rewards_ignore_formatting_only_differences():
    weights = train_bimodal.DEFAULT_REWARD_WEIGHTS
    raw = train_bimodal.reward_terms(r"\mbox{C}+x", "C+x", True, weights)
    assert raw["exact"] == 0.0 and raw["strict"] == 0.0
    standard = train_bimodal.reward_terms(r"\mbox{C}+x", "C+x", True, weights, standardize=True)
    assert standard["exact"] == standard["strict"] == 1.0 and standard["reward"] == pytest.approx(1.75)
    swapped = train_bimodal.reward_terms("A^{2}_{k}", "A_{k}^{2}", True, weights, standardize=True)
    assert swapped["strict"] == 1.0
    malformed = train_bimodal.reward_terms(r"\frac{1}{", "x", True, weights, standardize=True)
    assert malformed["exact"] == 0.0 and math.isfinite(malformed["reward"])
    assert train_bimodal.score_prediction(r"a\lt b", "a<b", True, standardize=True)["strict_exact"] == 1.0


def test_shard_dataset_filters_upsamples_and_standardizes(workspace):
    tokenizer = LatexTokenizer.load(workspace.shards / "tokenizer.json")
    kwargs = dict(stage="stage2", tokenizer=tokenizer, seed=5, allowed_sources=["crohme_train", "mathwriting_train"],
                  source_upsample={"crohme_train": 2.0, "mathwriting_synthetic": 9.0}, standardize_labels=True)
    dataset = train_bimodal.ResumableShardDataset(workspace.shards, batch_size=4, **kwargs)
    dataset.set_epoch(1)
    samples = list(dataset)
    counts = {source: sum(sample["source"] == source for sample in samples) for source in
              ("crohme_train", "mathwriting_train", "mathwriting_synthetic", "hme100k")}
    assert counts == {"crohme_train": 16, "mathwriting_train": 6, "mathwriting_synthetic": 0, "hme100k": 0}
    assert all(sample["normalized_latex"] == standardize_latex(normalize_training_latex(sample["latex"]))
               for sample in samples)
    skipped = train_bimodal.ResumableShardDataset(workspace.shards, batch_size=4, skip_batches=2, **kwargs)
    skipped.set_epoch(1)
    assert [sample["id"] for sample in skipped] == [sample["id"] for sample in samples][8:]
    downsampled = train_bimodal.ResumableShardDataset(
        workspace.shards, **{**kwargs, "source_upsample": {"crohme_train": 1.0, "mathwriting_train": 0.0}})
    assert {sample["source"] for sample in downsampled} == {"crohme_train"}
    batch = collate_bimodal_batch(samples[:4], tokenizer.pad_id)
    assert batch["is_anchor"].dtype == torch.bool and not batch["is_anchor"].any()
    assert not batch["has_stage3_role"].any()
    evaluation = train_bimodal.BimodalEvalDataset(workspace.shards, tokenizer=tokenizer,
                                                  allowed_sources=["crohme_val"], standardize_labels=True)
    assert len(evaluation) == 4 and {evaluation[index]["source"] for index in range(4)} == {"crohme_val"}


def test_rewards_score_case_structure_and_edits():
    weights = train_bimodal.DEFAULT_REWARD_WEIGHTS
    exact = train_bimodal.reward_terms(r"\frac{1}{2}", r"\frac{1}{2}", True, weights)
    assert exact["reward"] == pytest.approx(1.75) and exact["exact"] == exact["strict"] == 1.0
    case = train_bimodal.reward_terms("c+c", "C+c", True, weights)
    assert case["exact"] == case["strict"] == 0.0 and case["skeleton"] == 1.0
    assert case["reward"] == pytest.approx(0.5 - 0.5 / 3)
    braces = train_bimodal.reward_terms("x^2", "x^{2}", True, weights)
    assert braces["exact"] == 1.0 and braces["strict"] == 0.0
    unfinished = train_bimodal.reward_terms("x+1", "x+1", False, weights)
    assert unfinished["exact"] == 0.0 and unfinished["edit"] == 0.0
    flat = train_bimodal.reward_terms("12", r"\frac{1}{2}", True, weights)
    assert flat["skeleton"] < 1.0 and flat["edit"] > 0


def test_hf_tokens_are_explicit_only(monkeypatch, hub):
    from huggingface_hub import constants
    from huggingface_hub.utils import build_hf_headers

    def sends_token(headers):
        return any(key.lower() == "authorization" for key in headers)

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    monkeypatch.setenv("HF_TOKEN", "hf_xxxxxxxxxxxxxxxxxxxxxxxx")
    assert train_bimodal.configure_environment() is None
    assert "HF_TOKEN" not in os.environ and os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    assert train_bimodal.configure_environment(REAL_TOKEN) == REAL_TOKEN
    assert os.environ["HF_TOKEN"] == REAL_TOKEN and os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is True
    assert not sends_token(build_hf_headers())
    assert build_hf_headers(token=REAL_TOKEN)["authorization"] == f"Bearer {REAL_TOKEN}"
    train_bimodal.HFSync(Path("unused"), "me/runs", "runs/stage1", 20, REAL_TOKEN)
    assert hub.tokens == [REAL_TOKEN]
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    assert train_bimodal.redacted_argv(["x.py", "--hf-token", REAL_TOKEN, f"--hf-token={REAL_TOKEN}"]) == \
        ["x.py", "--hf-token", "***", "--hf-token=***"]


def test_structure_targets_feed_structure_loss():
    targets = {"stroke_group_ids": torch.tensor([0, 0, 1, 2]), "stroke_symbol_ids": torch.tensor([2, 2, 0, 1]),
               "relation_edges": torch.tensor([[0, 1, 1], [1, 2, 2]])}
    record = train_bimodal.structure_record(targets, 3, 3)
    assert record == {"stroke_groups": [0, 0, 1, 2], "symbol_classes": [2, 0, 1],
                      "relations": [(0, 1, 1), (1, 2, 2)], "relations_available": True}
    assert train_bimodal.structure_record({**targets, "stroke_symbol_ids": torch.tensor([2, 1, 0, 1])}, 3, 3) is None
    assert train_bimodal.structure_record({**targets, "relation_edges": torch.tensor([[0, 5, 1]])}, 3, 3) is None
    outputs = {"symbol": torch.randn(1, 4, 3, requires_grad=True), "group": torch.randn(1, 4, 4),
               "relation": torch.randn(1, 4, 4, 3), "count": torch.zeros(1, 3),
               "stroke_mask": torch.zeros(1, 4, dtype=torch.bool)}
    total, parts = structure_loss(outputs, [record], 3, {"symbol": 0.2, "group": 0.1, "relation": 0.1})
    assert torch.isfinite(total) and all(parts[key] > 0 for key in ("symbol", "group", "relation"))


def test_resumable_dataset_skips_exactly_the_consumed_batches(workspace):
    tokenizer = LatexTokenizer.load(workspace.shards / "tokenizer.json")
    kwargs = dict(stage="stage1", tokenizer=tokenizer, seed=3, use_upsample_hint=True, batch_size=4)
    full = train_bimodal.ResumableShardDataset(workspace.shards, **kwargs)
    full.set_epoch(1)
    ids = [sample["id"] for sample in full]
    skipped = train_bimodal.ResumableShardDataset(workspace.shards, skip_batches=3, **kwargs)
    skipped.set_epoch(1)
    rest = [sample["id"] for sample in skipped]
    assert len(ids) == 32 and len(rest) == len(ids) - 12
    assert sorted(ids[:12] + rest) == sorted(ids)


# --------------------------------------------------------------------------- end to end


def test_stage1_trains_vision_only_and_syncs_drive(stage1):
    output, drive = stage1.output, stage1.drive
    for name in ("best.pt", "recovery.pt", "metrics.jsonl", "status.json", "config_resolved.json"):
        assert (output / name).is_file(), name
        assert sha256_file(drive / name) == sha256_file(output / name), name
    status = read_json(output / "status.json")
    assert status["state"] == "completed" and status["stage"] == 1 and status["global_step"] == 16
    assert status["best_metric"]["normalized_exact"] >= 0
    metrics = read_metrics(output / "metrics.jsonl")
    assert [record["epoch"] for record in metrics] == [0, 1]
    assert [record["global_step"] for record in metrics] == [8, 16]
    assert all(record["train_samples"] == 32 for record in metrics)
    assert all(math.isfinite(record["train"]["loss"]) and record["train"]["r2l_ce"] > 0 for record in metrics)
    val = metrics[-1]["val"]
    assert val["modality"] == "vision" and set(val["by_source"]) == {"crohme", "mathwriting"}
    recovery = torch.load(output / "recovery.pt", map_location="cpu", weights_only=False)
    for key in ("stage", "epoch", "global_step", "elapsed_seconds", "model_config", "model_state_dict",
                "ema_state_dict", "optimizer_state_dict", "scheduler_state_dict", "best_metric", "tokenizer_sha256",
                "manifest_sha256"):
        assert key in recovery, key
    assert recovery["epoch"] == 2 and recovery["model_config"]["structure_symbols"] is None
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert best["weights"] == "ema" and best["stage"] == 1 and "model_state_dict" in best
    train_bimodal.seed_everything(1337)
    initial = BimodalMathModel(BimodalModelConfig.from_dict(recovery["model_config"])).state_dict()
    trained = recovery["model_state_dict"]
    stroke = [name for name in initial if name.startswith("stroke_encoder.")]
    assert stroke and all(torch.equal(initial[name], trained[name]) for name in stroke)
    assert not torch.equal(initial["decoder.token_embedding.weight"], trained["decoder.token_embedding.weight"])
    assert len(list(output.glob("epoch-*.pt"))) == 1


def test_wall_limit_pause_then_resume_from_drive(workspace, stage1):
    output, drive = workspace.root / "paused", workspace.root / "drive" / "paused"
    base = ["--config", workspace.config, "--stage", 1, "--shards", workspace.shards, "--output", output,
            "--drive-sync-dir", drive, *COMMON[:-2]]
    logs = run(*base, "--max-wall-minutes", 1e-6)
    status = read_json(drive / "status.json")
    assert "pausing" in logs and status["state"] == "paused_wall_limit"
    assert (status["epoch"], status["batch_index"], status["global_step"]) == (0, 1, 1)
    assert (drive / "recovery.pt").is_file()
    shutil.rmtree(output)
    logs = run(*base, "--max-wall-minutes", 0, "--resume", "auto")
    assert "restored run artifacts" in logs and "resuming after batch 1" in logs
    status = read_json(output / "status.json")
    reference = read_json(stage1.output / "status.json")
    assert status["state"] == "completed" and status["global_step"] == reference["global_step"]
    metrics = read_metrics(output / "metrics.jsonl")
    expected = read_metrics(stage1.output / "metrics.jsonl")
    assert [record["epoch"] for record in metrics] == [0, 1]
    assert [record["global_step"] for record in metrics] == [record["global_step"] for record in expected]
    assert [record["train_samples"] for record in metrics] == [32, 32]
    assert sha256_file(drive / "best.pt") == sha256_file(output / "best.pt")
    again = run(*base, "--max-wall-minutes", 0, "--resume", "auto")
    assert "already completed" in again


# --------------------------------------------------------------------------- Hugging Face dataset sync


def test_hf_sync_commits_changed_files_atomically_and_survives_outages(tmp_path, hub, capsys):
    output, prefix = tmp_path / "run", "runs/v1/stage1"
    output.mkdir()
    for name in train_bimodal.SYNC_FILES:
        (output / name).write_text(name, encoding="utf-8")
    sync = train_bimodal.HFSync(output, HF_REPO, prefix, 20, REAL_TOKEN)
    sync.ensure_repo()
    assert hub.repos == [{"repo_id": HF_REPO, "repo_type": "dataset", "private": True, "exist_ok": True}]
    assert hub.tokens == [REAL_TOKEN]
    assert sync.sync() and hub.commits == []
    assert sync.sync(force=True)
    assert [paths for _, paths in hub.commits] == [sorted(f"{prefix}/{name}" for name in train_bimodal.SYNC_FILES)]
    assert sync.sync(force=True) and len(hub.commits) == 1

    (output / "status.json").write_text('{"state": "running", "epoch": 3, "global_step": 9}', encoding="utf-8")
    sync.last -= 21 * 60
    assert sync.due() and sync.sync()
    assert hub.commits[-1] == (f"{prefix}: running epoch 3 step 9", [f"{prefix}/status.json"])

    (output / "metrics.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    hub.fail_commits, sync.last = 1, sync.last - 21 * 60
    capsys.readouterr()
    assert not sync.sync() and not sync.healthy and len(hub.commits) == 2
    assert "training continues" in capsys.readouterr().out
    hub.fail_commits = 3
    assert sync.sync(force=True) and sync.healthy and hub.fail_commits == 0
    assert hub.commits[-1][1] == [f"{prefix}/metrics.jsonl"] and "retrying" in capsys.readouterr().out
    (output / "recovery.pt").write_text("newer", encoding="utf-8")
    hub.fail_commits = 10
    assert not sync.sync(force=True) and hub.fail_commits == 6 and not sync.healthy
    assert "ERROR" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        train_bimodal.HFSync(output, HF_REPO, prefix, 20, None)


def test_hf_resume_restores_run_and_stage2_inits_from_hf(workspace, stage1, hub, monkeypatch):
    output, prefix = workspace.root / "hf_stage1", "runs/v1/stage1"
    base = ["--config", workspace.config, "--stage", 1, "--shards", workspace.shards, "--output", output,
            "--hf-sync-repo", HF_REPO, "--hf-sync-prefix", prefix, "--hf-squash-history", *COMMON]
    with monkeypatch.context() as patch:
        patch.setattr(train_bimodal.Trainer, "wall_reached", lambda self: self.global_step >= 9)
        logs = run(*base)
    assert "pausing" in logs and hub.squashes == 1
    assert hub.repos == [{"repo_id": HF_REPO, "repo_type": "dataset", "private": True, "exist_ok": True}]
    uploaded = [paths for _, paths in hub.commits]
    assert uploaded[0] == [f"{prefix}/config_resolved.json", f"{prefix}/status.json"]
    assert {f"{prefix}/best.pt", f"{prefix}/metrics.jsonl", f"{prefix}/recovery.pt"} <= set(uploaded[1])
    assert sum(f"{prefix}/config_resolved.json" in paths for paths in uploaded) == 1
    assert json.loads(hub.files[f"{prefix}/status.json"])["state"] == "paused_wall_limit"
    assert REAL_TOKEN.encode() not in hub.files[f"{prefix}/config_resolved.json"]

    shutil.rmtree(output)
    commits_before = len(hub.commits)
    logs = run(*base, "--resume", "auto")
    assert f"restored run artifacts from https://huggingface.co/datasets/{HF_REPO}/tree/main/{prefix}" in logs
    assert "resuming after batch 1" in logs
    assert set(hub.downloads) == {f"{prefix}/{name}" for name in train_bimodal.SYNC_FILES}
    assert f"{prefix}/recovery.pt" not in hub.commits[commits_before][1]
    status = read_json(output / "status.json")
    assert status["state"] == "completed" and status["global_step"] == read_json(stage1.output / "status.json")[
        "global_step"]
    assert [record["epoch"] for record in read_metrics(output / "metrics.jsonl")] == [0, 1]
    assert json.loads(hub.files[f"{prefix}/status.json"])["state"] == "completed"
    assert hub.files[f"{prefix}/best.pt"] == (output / "best.pt").read_bytes()

    stage2 = workspace.root / "hf_stage2"
    logs = run("--config", workspace.config, "--stage", 2, "--shards", workspace.shards, "--output", stage2,
               "--hf-sync-repo", HF_REPO, "--hf-sync-prefix", "runs/v1/stage2", "--hf-init-path", f"{prefix}/best.pt",
               "--max-steps", 1, *COMMON)
    assert f"init-from: downloaded hf://datasets/{HF_REPO}/{prefix}/best.pt" in logs
    assert f"init-from {stage2 / '_init_from.pt'}" in logs and "freshly initialized" in logs
    assert (stage2 / "_init_from.pt").read_bytes() == hub.files[f"{prefix}/best.pt"]
    assert "runs/v1/stage2/best.pt" in hub.files
    assert read_json(stage2 / "config_resolved.json")["init_from"] == str(stage2 / "_init_from.pt")
    with pytest.raises(SystemExit, match="cannot download runs/v1/stage9/best.pt"):
        run("--config", workspace.config, "--stage", 2, "--shards", workspace.shards,
            "--output", workspace.root / "hf_missing", "--hf-sync-repo", HF_REPO,
            "--hf-init-path", "runs/v1/stage9/best.pt", *COMMON)


def test_hf_sync_repo_fails_fast_without_token(workspace, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        run("--config", workspace.config, "--stage", 1, "--shards", workspace.shards,
            "--output", workspace.root / "hf_no_token", "--hf-sync-repo", HF_REPO, *COMMON)
    with pytest.raises(SystemExit):
        train_bimodal.parse_args(["--shards", "s", "--stage", "1", "--output", "o", "--hf-init-path", "x/best.pt"])
    args = train_bimodal.parse_args(["--shards", "s", "--stage", "2", "--output", "o", "--hf-sync-repo", HF_REPO])
    assert args.hf_sync_prefix == "runs/stage2"


def test_grad_accum_flag_overrides_config_and_schedule(workspace, tmp_path):
    config = yaml.safe_load(workspace.config.read_text(encoding="utf-8"))
    base = ["--config", str(workspace.config), "--stage", "2", "--shards", str(workspace.shards), *COMMON]

    def trainer(*extra: str) -> train_bimodal.Trainer:
        return train_bimodal.Trainer(train_bimodal.parse_args([*base, "--output", str(tmp_path / "run"), *extra]),
                                     config)

    default = trainer()
    assert train_bimodal.parse_args(base + ["--output", "o"]).grad_accum is None
    assert int(default.cfg.get("grad_accum", 1)) == 1 and default.cfg["batch_size"] == 4
    assert default.build_schedule().main_steps == 2 * math.ceil(24 / 4)
    accumulated = trainer("--batch-size", "5", "--grad-accum", "2")
    assert accumulated.cfg["grad_accum"] == 2 and accumulated.cfg["batch_size"] == 5
    assert accumulated.build_schedule().main_steps == 2 * math.ceil(math.ceil(24 / 5) / 2) == 6
    assert config["stage2"].get("grad_accum", 1) == 1
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        train_bimodal.parse_args([*base, "--output", "o", "--grad-accum", "0"])


def test_grad_accum_flag_accumulates_micro_batches(workspace):
    output = workspace.root / "grad_accum"
    run("--config", workspace.config, "--stage", 1, "--shards", workspace.shards, "--output", output,
        "--grad-accum", 2, "--max-steps", 2, *COMMON)
    status = read_json(output / "status.json")
    assert status["global_step"] == 2 and status["batch_index"] == 4
    assert read_json(output / "config_resolved.json")["stage_config"]["grad_accum"] == 2
    assert read_metrics(output / "metrics.jsonl")[-1]["train_samples"] == 4 * 4
    recovery = torch.load(output / "recovery.pt", map_location="cpu", weights_only=False)
    assert recovery["grad_accum"] == 2 and recovery["scheduler_state_dict"]["main_steps"] == 2


def test_stage2_bimodal_from_stage1_with_structure_r2l_and_cooldown(stage2, stage1):
    logs = stage2.logs
    assert f"init-from {stage1.output / 'best.pt'}" in logs and "shape mismatch, freshly initialized" in logs
    assert "stroke_encoder.structure" in logs
    best = torch.load(stage2.output / "best.pt", map_location="cpu", weights_only=False)
    assert best["model_config"]["structure_symbols"] == SYMBOLS
    assert best["model_config"]["structure_relations"] == RELATIONS
    metrics = read_metrics(stage2.output / "metrics.jsonl")
    assert [record["cooldown"] for record in metrics] == [False, False, True]
    assert [record["train_samples"] for record in metrics] == [24, 24, 18]
    assert all(record["train"]["r2l_ce"] > 0 and math.isfinite(record["train"]["loss"]) for record in metrics)
    structured = [record["train"] for record in metrics if "structure" in record["train"]]
    assert structured and all(train["structure"] > 0 and "structure_symbol" in train for train in structured)
    assert all("structure_misaligned" not in record["train"] for record in metrics)
    assert all(0 < record["train"]["ink_rows"] <= 1 for record in metrics)
    assert pytest.approx(3e-5) in metrics[-1]["lr"]
    assert metrics[-1]["val"]["modality"] == "auto"
    status = read_json(stage2.output / "status.json")
    assert status["state"] == "completed" and status["global_step"] == 6 + 6 + 5


def test_stage26_coupled_augmentation_and_teacher_kd_end_to_end(workspace, stage1, tmp_path):
    tokenizer = LatexTokenizer.build(normalize_training_latex(label) for label in LABELS[:6])
    teacher_path = tmp_path / "teacher.pt"
    save_checkpoint(teacher_path, tiny_teacher(tokenizer), tokenizer,
                    PreprocessingConfig(min_resampled_points_per_stroke=8), 0, {})
    config = yaml.safe_load(workspace.config.read_text(encoding="utf-8"))
    stage26 = yaml.safe_load(STAGE26_CONFIG.read_text(encoding="utf-8"))["stage2"]
    config["stage2"].update(epochs=2, cooldown_epochs=1, teacher_checkpoint=str(tmp_path / "missing.pt"),
                            teacher_kd_weight=0.5, cooldown_teacher_kd_weight=0.2, teacher_kd_temp=2.0,
                            teacher_kd_min_target_prob=0.0, cooldown_disable_band_mask=True,
                            augment={**stage26["augment"], "point_dropout": 0.2})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "stage26"
    logs = run("--config", path, "--stage", 2, "--shards", workspace.shards, "--output", output,
               "--init-from", stage1.output / "best.pt", "--teacher-checkpoint", teacher_path, *COMMON)
    assert f"teacher: {teacher_path}" in logs and "kd_loss=" in logs and "kd_coverage=" in logs
    resolved = read_json(output / "config_resolved.json")
    assert resolved["teacher_checkpoint"] == str(teacher_path) == resolved["stage_config"]["teacher_checkpoint"]
    assert resolved["teacher"]["normalization"] == "expression" and resolved["teacher"]["shared_tokens"] > 20
    assert resolved["stage_config"]["augment"]["coupled_geometry"] is True
    metrics = read_metrics(output / "metrics.jsonl")
    assert [record["cooldown"] for record in metrics] == [False, True]
    for record in metrics:
        train = record["train"]
        assert train["kd_loss"] > 0 and 0 < train["kd_coverage"] <= 1 and train["kd_rows"] > 0
        assert math.isfinite(train["loss"]) and "structure_misaligned" not in train
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model = BimodalMathModel(BimodalModelConfig.from_dict({**best["model_config"], "vision_pretrained": False}))
    assert set(best["model_state_dict"]) == set(model.state_dict())
    model.load_state_dict(best["model_state_dict"])
    recovery = torch.load(output / "recovery.pt", map_location="cpu", weights_only=False)
    assert recovery["train_config"]["teacher_checkpoint"] == str(teacher_path)


def test_mine_hard_then_grpo_stage3(workspace, stage3):
    hard = read_json(workspace.shards / "hard_manifest.json")
    assert hard["hard"] == 6 and hard["anchors"] == 4 and hard["samples"] == 10 and hard["scanned"] == 24
    roles = []
    for shard in hard["shards"]:
        path = workspace.shards / "stage3_hard" / shard["file"]
        assert sha256_file(path) == shard["sha256"]
        for record in iter_shard(path):
            meta = json.loads(record["json"])
            roles.append(meta["stage3_role"])
            assert (record["npz"] is not None) == meta["has_strokes"]
            if meta["stage3_role"] == "anchor":
                assert not meta["synthetic"]
            else:
                assert "prediction" in meta["mined"]
    assert sorted(roles) == ["anchor"] * 4 + ["hard"] * 6

    metrics = read_metrics(stage3.output / "metrics.jsonl")
    assert len(metrics) == 1
    train = metrics[0]["train"]
    assert all(math.isfinite(train[key]) for key in ("loss", "policy_loss", "kl", "reward", "anchor_ce"))
    assert train["reward_std"] > 0 and train["varied_groups"] > 0
    assert train["kl"] > 0 and train["anchor_ce"] > 0 and train["anchor_rows"] >= 1
    status = read_json(stage3.output / "status.json")
    assert status["state"] == "completed" and status["global_step"] == 3
    config = read_json(stage3.output / "config_resolved.json")
    assert config["reference_checkpoint"].endswith("best.pt")
    best = torch.load(stage3.output / "best.pt", map_location="cpu", weights_only=False)
    assert best["stage"] == 3 and best["model_config"]["structure_symbols"] == SYMBOLS


def test_stage25_then_fixed_grpo_stage3(workspace, stage2, monkeypatch):
    root = workspace.root / "stage25"
    shards = root / "shards"
    shutil.copytree(workspace.shards, shards, ignore=shutil.ignore_patterns("stage3_hard", "hard_manifest.json"))
    config = yaml.safe_load(workspace.config.read_text(encoding="utf-8"))
    config["standardize_labels"] = True
    config["stage2"].update(epochs=2, cooldown_epochs=1, cooldown_lr=1e-4, stroke_lr=3e-3, min_lr_ratio=0.1,
                            allowed_sources=["crohme_train", "mathwriting_train"],
                            source_upsample={"crohme_train": 2.0, "mathwriting_train": 1.0},
                            vision_drop_prob=0.5, stroke_drop_prob=0.05, cooldown_vision_drop_prob=0.15)
    config["stage3"].update(freeze_vision=True, stroke_lr=1e-3, vision_lr=0.0, min_lr_ratio=0.1,
                            sft_anchor_weight=0.25, r2l_anchor_weight=0.05)
    config["validation"].update(eval_every_steps=3, log_modalities=True)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    polish = root / "run25"
    logs = run("--config", path, "--stage", 2, "--shards", shards, "--output", polish,
               "--init-from", stage2.output / "best.pt", *COMMON)
    loaded = re.search(r"loaded (\d+)/(\d+) tensors", logs)
    assert loaded and loaded[1] == loaded[2]
    assert "stroke_decay lr=3.0e-03" in logs
    metrics = read_metrics(polish / "metrics.jsonl")
    epochs = [record for record in metrics if record["type"] == "epoch"]
    steps = [record for record in metrics if record["type"] == "step_val"]
    assert [record["train_samples"] for record in epochs] == [22, 22]
    assert [record["global_step"] for record in steps] == [3, 6, 9, 12]
    for record in steps + epochs:
        assert set(record["val"]["modalities"]) == {"vision", "stroke"}
        assert 0 <= record["val"]["modalities"]["stroke"]["normalized_exact"] <= 1
    assert epochs[-1]["val"] == steps[-1]["val"]
    resolved = read_json(polish / "config_resolved.json")
    assert resolved["standardize_labels"] is True and resolved["schedule"]["floor"] == pytest.approx(0.1)
    assert resolved["init_report"]["loaded"] == resolved["init_report"]["total"]

    run("--config", path, "--mine-hard", "--shards", shards, "--init-from", polish / "best.pt",
        "--mine-sources", "crohme_train", "mathwriting_train", "--crohme-upsample", 3, "--mine-limit", 6,
        "--anchor-limit", 4, "--device", "cpu", "--batch-size", 8)
    hard = read_json(shards / "hard_manifest.json")
    assert hard["scanned"] == 14 and hard["standardize_labels"] is True
    assert set(hard["sources"]) <= {"crohme_train", "mathwriting_train"}
    assert hard["unique_samples"] == 10 and hard["anchors"] == 4
    crohme_hard = []
    for shard in hard["shards"]:
        for record in iter_shard(shards / "stage3_hard" / shard["file"]):
            meta = json.loads(record["json"])
            if meta["stage3_role"] == "hard" and meta["source"] == "crohme_train":
                crohme_hard.append(meta["id"])
    assert all(crohme_hard.count(identifier) == 3 for identifier in crohme_hard)
    assert hard["samples"] == 10 + 2 * len(set(crohme_hard)) == hard["hard"] + hard["anchors"]

    calls = []
    original = train_bimodal.Trainer.grpo_step

    def recording(self, batch, micro):
        loss, stats = original(self, batch, micro)
        calls.append((int(batch["is_anchor"].sum()), int((~batch["synthetic"]).sum()),
                      bool(batch["has_stage3_role"].all()), dict(stats)))
        return loss, stats

    monkeypatch.setattr(train_bimodal.Trainer, "grpo_step", recording)
    grpo = root / "run3"
    run("--config", path, "--stage", 3, "--shards", shards, "--output", grpo, "--init-from", polish / "best.pt",
        "--eval-every-steps", 0, *COMMON)
    assert calls and all(roles for _, _, roles, _ in calls)
    assert any(anchors for anchors, _, _, _ in calls)
    assert any(humans > anchors for anchors, humans, _, _ in calls)
    for anchors, _, _, stats in calls:
        if anchors:
            assert stats["anchor_rows"] == anchors and stats["anchor_ce"] > 0 and stats["r2l_anchor_ce"] > 0
        else:
            assert "anchor_rows" not in stats and "r2l_anchor_ce" not in stats
    init = torch.load(polish / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    recovery = torch.load(grpo / "recovery.pt", map_location="cpu", weights_only=False)
    trained = recovery["model_state_dict"]
    vision = [name for name in init if name.startswith("vision_encoder.")]
    assert vision and all(torch.equal(init[name], trained[name]) for name in vision)
    assert any(not torch.equal(init[name], trained[name]) for name in trained if name.startswith("decoder."))
    assert recovery["scheduler_state_dict"]["floor"] == pytest.approx(0.1)
    assert [group["name"] for group in recovery["optimizer_state_dict"]["param_groups"]] == \
        ["main_no_decay", "main_decay", "stroke_no_decay", "stroke_decay"]
    train = read_metrics(grpo / "metrics.jsonl")[-1]["train"]
    assert train["r2l_anchor_ce"] > 0 and not any(record["type"] == "step_val"
                                                   for record in read_metrics(grpo / "metrics.jsonl"))


def test_eval_only_reports_per_source_metrics(workspace, stage1, stage2):
    output = workspace.root / "eval"
    printed = run("--config", workspace.config, "--eval-only", "--shards", workspace.shards,
                  "--init-from", stage2.output / "best.pt", "--output", output, "--device", "cpu")
    metrics = read_json(output / "eval_metrics.json")
    assert json.loads(printed[printed.index("{"):])["samples"] == metrics["samples"] == 8
    assert metrics["modality"] == "auto" and metrics["stage"] == 2
    assert {name: group["samples"] for name, group in metrics["by_source"].items()} == {"crohme": 4,
                                                                                        "mathwriting": 4}
    for group in (metrics, *metrics["by_source"].values()):
        assert 0 <= group["normalized_exact"] <= 1 and 0 <= group["strict_exact"] <= 1
        assert group["token_cer"] >= 0 and group["avg_tokens"] >= 0
    assert len((output / "eval_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 8
    run("--config", workspace.config, "--eval-only", "--shards", workspace.shards,
        "--init-from", stage1.output / "best.pt", "--output", output, "--device", "cpu")
    assert read_json(output / "eval_metrics.json")["modality"] == "vision"


# --------------------------------------------------------------------------- Drive IDs and Colab notebook


DRIVEFS_COLUMNS = ("stable_id INTEGER PRIMARY KEY, id TEXT, proto BLOB, trashed BOOLEAN, starred BOOLEAN, "
                   "is_owner BOOLEAN, mime_type TEXT, is_folder BOOLEAN, modified_date INTEGER, "
                   "shared_with_me_date INTEGER, viewed_by_me_date INTEGER, file_size INTEGER, is_tombstone BOOLEAN, "
                   "local_title TEXT, subscribed BOOLEAN, team_drive_stable_id INTEGER, "
                   "inaccessible_inheritance_broken BOOLEAN")


def drivefs_db(path: Path) -> sqlite3.Connection:
    """DriveFS schema in WAL mode; the open connection keeps recent rows only in -wal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute(f"CREATE TABLE items ({DRIVEFS_COLUMNS})")
    connection.execute("CREATE TABLE stable_parents (item_stable_id INTEGER, parent_stable_id INTEGER, "
                       "local_title_hash INTEGER)")
    rows = [  # stable_id, parent, id, title, folder, size, trashed, tombstone
        (1, 0, "root-folder-id-000000000", "My Drive", 1, 0, 0, 0),
        (10, 1, "colab-folder-id-00000000", "math-recogniser-colab", 1, 0, 0, 0),
        (11, 1, "old-colab-folder-0000000", "math-recogniser-colab", 1, 0, 1, 0),
        (12, 11, "old-zip-id-0000000000000", "math-recogniser.zip", 0, 5, 0, 0),
        (20, 10, "zip-id-000000000000000000", "math-recogniser.zip", 0, 679856, 0, 0),
        (21, 10, "ids-json-id-0000000000000", "drive_file_ids.json", 0, 900, 0, 0),
        (22, 10, "notebook-id-0000000000000", "train_bimodal_g4.ipynb", 0, 21196, 0, 0),
        (30, 10, "shards-folder-id-00000000", "shards", 1, 0, 0, 0),
        (31, 30, "manifest-id-0000000000000", "manifest.json", 0, 112257, 0, 0),
        (32, 30, "tokenizer-id-000000000000", "tokenizer.json", 0, 14959, 0, 0),
        (40, 30, "stage1-folder-id-00000000", "stage1_train", 1, 0, 0, 0),
        (41, 40, "s1-shard0-id-000000000000", "shard-00000.tar.gz", 0, 1000, 0, 0),
        (42, 40, "s1-shard1-id-000000000000", "shard-00001.tar.gz", 0, 1001, 0, 0),
        (43, 40, "s1-trashed-id-00000000000", "shard-00002.tar.gz", 0, 1002, 1, 0),
        (44, 40, "s1-tomb-id-00000000000000", "shard-00003.tar.gz", 0, 1003, 0, 1),
        (45, 40, "local-123456", "shard-00004.tar.gz", 0, 1004, 0, 0),
        (46, 40, None, "shard-00005.tar.gz", 0, 1005, 0, 0),
        (50, 30, "val-folder-id-00000000000", "val_dev", 1, 0, 0, 0),
        (51, 50, "val-shard0-id-00000000000", "shard-00000.tar.gz", 0, 2000, 0, 0),
        (60, 30, "trashed-folder-id-0000000", "old", 1, 0, 1, 0),
        (61, 60, "hidden-by-folder-00000000", "shard-99999.tar.gz", 0, 1, 0, 0),
        (70, 1, "elsewhere-id-000000000000", "unrelated.txt", 0, 3, 0, 0),
    ]
    for stable_id, parent, identifier, title, folder, size, trashed, tombstone in rows:
        connection.execute("INSERT INTO items (stable_id, id, trashed, is_folder, file_size, is_tombstone, "
                           "local_title) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (stable_id, identifier, trashed, folder, size, tombstone, title))
        if parent:
            connection.execute("INSERT INTO stable_parents VALUES (?, ?, 0)", (stable_id, parent))
    connection.commit()
    return connection


def test_export_drive_ids_walks_live_cloud_files(tmp_path, monkeypatch):
    db = tmp_path / "Google" / "DriveFS" / "1038" / "metadata_sqlite_db"
    connection = drivefs_db(db)
    try:
        assert db.with_name(db.name + "-wal").stat().st_size > 0
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert export_drive_ids.find_db() == db
        export, pending = export_drive_ids.export_ids(db)
    finally:
        connection.close()
    assert export["format"] == "drive-file-ids-v1" and export["root_folder"] == "math-recogniser-colab"
    assert export["root_folder_id"] == "colab-folder-id-00000000"
    assert export["files"] == {
        "drive_file_ids.json": {"id": "ids-json-id-0000000000000", "size": 900},
        "math-recogniser.zip": {"id": "zip-id-000000000000000000", "size": 679856},
        "shards/manifest.json": {"id": "manifest-id-0000000000000", "size": 112257},
        "shards/stage1_train/shard-00000.tar.gz": {"id": "s1-shard0-id-000000000000", "size": 1000},
        "shards/stage1_train/shard-00001.tar.gz": {"id": "s1-shard1-id-000000000000", "size": 1001},
        "shards/tokenizer.json": {"id": "tokenizer-id-000000000000", "size": 14959},
        "shards/val_dev/shard-00000.tar.gz": {"id": "val-shard0-id-00000000000", "size": 2000},
        "train_bimodal_g4.ipynb": {"id": "notebook-id-0000000000000", "size": 21196},
    }
    assert export["counts"] == {"total": 8, "by_dir": {".": 3, "shards": 2, "shards/stage1_train": 2,
                                                       "shards/val_dev": 1}}
    assert pending == ["shards/stage1_train/shard-00004.tar.gz", "shards/stage1_train/shard-00005.tar.gz"]


def test_export_drive_ids_cli_embeds_notebook_and_packs_zip_in_place(tmp_path):
    db = tmp_path / "drivefs" / "metadata_sqlite_db"
    connection = drivefs_db(db)
    drive = tmp_path / "drive"
    (drive / "shards").mkdir(parents=True)
    (drive / "shards" / "audit_report.json").write_text("{}", encoding="utf-8")
    (drive / "math-recogniser.zip").write_bytes(b"old")
    file_id = (drive / "math-recogniser.zip").stat().st_ino
    notebook = tmp_path / "notebook.ipynb"
    shutil.copyfile(NOTEBOOK, notebook)
    ml = tmp_path / "ml"
    for name in ("train_bimodal.py", "data/bimodal_shards.py", "colab/train_bimodal_g4.ipynb", "tools/export.py",
                 ".venv/lib.py", "data/__pycache__/x.pyc", "checkpoints/a.py", "datasets/b.py", "best.pt",
                 "runs/metrics.jsonl", "sample.png"):
        (ml / name).parent.mkdir(parents=True, exist_ok=True)
        (ml / name).write_text("x", encoding="utf-8")
    original_pack = export_drive_ids.pack_zip
    try:
        export_drive_ids.pack_zip = lambda target: original_pack(target, ml)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for _ in range(2):
                export_drive_ids.main(["--db", str(db), "--drive-dir", str(drive), "--pack-zip",
                                       "--embed-notebook", str(notebook)])
    finally:
        export_drive_ids.pack_zip = original_pack
        connection.close()
    logs = output.getvalue()
    exported = read_json(drive / "drive_file_ids.json")
    archive = drive / "math-recogniser.zip"
    assert archive.stat().st_ino == file_id
    with zipfile.ZipFile(archive) as packed:
        assert sorted(packed.namelist()) == ["math-recogniser/ml/colab/train_bimodal_g4.ipynb",
                                             "math-recogniser/ml/data/bimodal_shards.py",
                                             "math-recogniser/ml/tools/export.py",
                                             "math-recogniser/ml/train_bimodal.py"]
    assert exported["files"]["math-recogniser.zip"]["size"] == archive.stat().st_size
    assert re.search(r"shards/stage1_train\s+2\s+shard-00004\.tar\.gz, shard-00005\.tar\.gz", logs)
    assert re.search(r"shards\s+1\s+audit_report\.json", logs)
    assert "zip-id-000000000000000000" in logs and "ids-json-id-0000000000000" in logs

    cell = notebook_cells(notebook)[1]
    assert cell.count("BEGIN EMBEDDED_DRIVE_IDS") == 1 and 'DRIVE_IDS_FILE_ID = "ids-json-id-0000000000000"' in cell
    embedded = embedded_drive_ids(cell)
    assert set(embedded) == set(exported["files"]) - {"drive_file_ids.json", "train_bimodal_g4.ipynb"}
    assert embedded["shards/val_dev/shard-00000.tar.gz"] == {"id": "val-shard0-id-00000000000", "size": 2000}


def notebook_cells(path: Path = NOTEBOOK) -> list[str]:
    return ["".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def embedded_drive_ids(source: str) -> dict:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "EMBEDDED_DRIVE_IDS":
            return ast.literal_eval(node.value)
    raise AssertionError("EMBEDDED_DRIVE_IDS not found")


STORAGE_MODES = frozenset({"hf", "drive_public_hf", "drive_mount"})


def mount_modes(source: str) -> list[frozenset]:
    """STORAGE_MODE values under which each Drive mount call or ``google.colab.drive`` import can run."""
    found = []

    def branch(test):
        if not (isinstance(test, ast.Compare) and getattr(test.left, "id", None) == "STORAGE_MODE"):
            return None
        node, op = test.comparators[0], test.ops[0]
        values = frozenset([node.value] if isinstance(node, ast.Constant)
                           else [element.value for element in getattr(node, "elts", [])])
        if not values <= STORAGE_MODES:
            return None
        if isinstance(op, (ast.Eq, ast.In)):
            return values, STORAGE_MODES - values
        if isinstance(op, (ast.NotEq, ast.NotIn)):
            return STORAGE_MODES - values, values
        return None

    def visit(node, modes):
        if isinstance(node, ast.If) and (split := branch(node.test)):
            for child in node.body:
                visit(child, modes & split[0])
            for child in node.orelse:
                visit(child, modes & split[1])
            return
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in ("mount", "flush_and_unmount"):
            found.append(modes)
        if isinstance(node, ast.ImportFrom) and node.module == "google.colab" and \
                any(alias.name == "drive" for alias in node.names):
            found.append(modes)
        for child in ast.iter_child_nodes(node):
            visit(child, modes)

    visit(ast.parse(source), STORAGE_MODES)
    return found


def test_colab_notebook_is_valid():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4 and notebook["nbformat_minor"] >= 5
    cells = notebook["cells"]
    header = "".join(cells[0]["source"])
    assert cells[0]["cell_type"] == "markdown" and "100" in header
    for needle in ("HF_TOKEN", "Secrets", "Notebook access", "HF_REPO_ID", "HF_SHARDS_REPO", "G4 (RTX PRO 6000)",
                   "Run all", 'STORAGE_MODE = "hf"', "pack_stage2_colab_cpu.ipynb", '"--grad-accum", "2"'):
        assert needle in header, needle
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 5 and all(cell.get("id") for cell in cells)
    sources = ["".join(cell["source"]) for cell in code]
    for source in sources:
        ast.parse(source)
        assert all(not line.lstrip().startswith(("%", "!")) for line in source.splitlines())
    text = "\n".join(sources)
    assert re.search(r'^STORAGE_MODE = "hf" ', sources[0], re.M)
    assert re.search(r'^HF_REPO_ID = "Wosca/math-recogniser-runs" ', sources[0], re.M)
    assert re.search(r'^HF_SHARDS_REPO = "Wosca/math-recogniser-shards" ', sources[0], re.M)
    for needle in ("STAGE = 1", "MAX_WALL_MINUTES = 180", "AUTO_DISCONNECT = True", 'RUN_NAME = "bimodal_75m_v1"',
                   "COMPILE = True", "STAGE3_COPY_STAGE2 = False", "EXTRA_ARGS = []", '"drive_public_hf"',
                   '"drive_mount"', "DRIVE_IDS_FILE_ID = ", 'userdata.get("HF_TOKEN")', "is_bf16_supported",
                   "nvidia-smi", "drive.usercontent.google.com/download?id={}&export=download&confirm=t",
                   r'b"PK\x03\x04"', r'b"\x1f\x8b"', "ThreadPoolExecutor(max_workers=8)", "verify_manifest_checksums",
                   "hard_manifest.json", "train_bimodal.py", "--hf-sync-repo", "--hf-sync-prefix", "--hf-init-path",
                   "--resume", "auto", "--max-wall-minutes", "--drive-sync-dir", "--sync-every-minutes", "--compile",
                   "--init-from", "TRAIN_EXIT_CODE", "runtime.unassign()", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
                   "antlr4-python3-runtime", "huggingface_hub", "huggingface.co/datasets/",
                   'hf_hub_download(repo_id=HF_SHARDS_REPO, filename="math-recogniser.zip"', "force_download=True",
                   "list_repo_files(HF_SHARDS_REPO", "TRAIN_SOURCE", '"--grad-accum"'):
        assert needle in text, needle
    assert "install torch" not in text.lower()
    modes = [mode for source in sources for mode in mount_modes(source)]
    assert len(modes) >= 3 and all(mode == {"drive_mount"} for mode in modes), modes


def test_colab_notebook_embeds_all_stage1_drive_ids():
    embedded = embedded_drive_ids(notebook_cells()[1])
    for name in ("math-recogniser.zip", "shards/manifest.json", "shards/tokenizer.json",
                 "shards/val_dev/shard-00000.tar.gz"):
        assert name in embedded, name
    stage1 = sorted(name for name in embedded if name.startswith("shards/stage1_train/"))
    assert stage1 == [f"shards/stage1_train/shard-{index:05d}.tar.gz" for index in range(122)]
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{25,}", entry["id"]) and entry["size"] > 0 for entry in embedded.values())


@pytest.mark.parametrize("mode", ["hf", "drive_public_hf", "drive_mount"])
def test_colab_notebook_cell1_storage_modes(mode, tmp_path, monkeypatch):
    mounted = []
    colab = ModuleType("google.colab")
    colab.userdata = SimpleNamespace(get=lambda key: REAL_TOKEN if key == "HF_TOKEN" else None)
    colab.drive = SimpleNamespace(mount=mounted.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="A100-SXM4-40GB"))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(name="A100-SXM4-40GB", total_memory=40 * 2**30))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    source = notebook_cells()[1].replace('STORAGE_MODE = "hf" ', f'STORAGE_MODE = "{mode}" ', 1)
    source = source.replace('"/content/drive/MyDrive/math-recogniser-colab"', repr(str(tmp_path / "drive")))

    def execute(code: str) -> tuple[dict, str]:
        namespace: dict = {}
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            exec(compile(code, "cell-1", "exec"), namespace)
        return namespace, printed.getvalue()

    if mode != "drive_mount":
        with pytest.raises(SystemExit, match="HF_REPO_ID"):
            execute(source.replace('HF_REPO_ID = "Wosca/math-recogniser-runs"', 'HF_REPO_ID = ""'))
    blank_shards = source.replace('HF_SHARDS_REPO = "Wosca/math-recogniser-shards"', 'HF_SHARDS_REPO = ""')
    if mode == "hf":
        with pytest.raises(SystemExit, match="HF_SHARDS_REPO"):
            execute(blank_shards)
    elif mode == "drive_public_hf":
        execute(blank_shards)
    hint = '["--batch-size", "48", "--grad-accum", "2"]'
    namespace, printed = execute(source)
    assert hint not in printed and namespace["VRAM_GIB"] == 40
    stage2 = source.replace("STAGE = 1 ", "STAGE = 2 ", 1)
    namespace, printed = execute(stage2)
    assert "HINT" in printed and hint in printed
    _, printed = execute(stage2.replace("EXTRA_ARGS = []", f"EXTRA_ARGS = {hint}"))
    assert hint not in printed
    if mode == "drive_mount":
        assert len(mounted) == 3 and set(mounted) == {"/content/drive"} and "HF_TOKEN" not in os.environ
        assert (tmp_path / "drive" / "runs" / "bimodal_75m_v1" / "stage2").is_dir()
    else:
        assert mounted == [] and namespace["HF_TOKEN"] == REAL_TOKEN
        assert os.environ["HF_TOKEN"] == REAL_TOKEN and os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert namespace["HF_PREFIX"] == "runs/bimodal_75m_v1/stage2"
        assert namespace["HF_SHARDS_REPO"] == "Wosca/math-recogniser-shards"
    assert "math-recogniser.zip" in namespace["EMBEDDED_DRIVE_IDS"]


def test_colab_notebook_hf_mode_stages_manifest_shards(workspace, hub, tmp_path):
    shards = workspace.shards
    manifest = read_json(shards / "manifest.json")
    for name in ("manifest.json", "tokenizer.json"):
        hub.files[f"shards/{name}"] = (shards / name).read_bytes()
    for directory in ("stage2_train", "val_dev"):
        for path in sorted((shards / directory).glob("shard-*.tar")):
            hub.files[f"shards/{directory}/{path.name}.gz"] = gzip.compress(path.read_bytes(), compresslevel=1)
    hub.files["shards/stage2_train/shard-00099.tar.gz"] = gzip.compress(b"stale shard from an older pack")
    local = tmp_path / "content" / "shards"
    namespace = {"STAGE": 2, "STAGE3_COPY_STAGE2": False, "STORAGE_MODE": "hf", "HF_TOKEN": REAL_TOKEN,
                 "HF_SHARDS_REPO": "Wosca/math-recogniser-shards", "LOCAL_SHARDS": local,
                 "DRIVE_SHARDS": tmp_path / "drive", "DRIVE_IDS": {}, "json": json, "os": os, "shutil": shutil,
                 "time": time, "Path": Path}
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        exec(compile(notebook_cells()[3], "cell-3", "exec"), namespace)
    expected = {f"{info['directory']}/{shard['file']}" for stage, info in manifest["stages"].items()
                if stage in ("stage2", "val") for shard in info["shards"]}
    staged = {path.relative_to(local).as_posix() for path in local.rglob("*") if path.is_file()}
    assert staged == expected | {"manifest.json", "tokenizer.json"}
    assert "shard-00099" not in printed.getvalue() and "verified" in printed.getvalue()
    assert (local / "tokenizer.json").read_bytes() == (shards / "tokenizer.json").read_bytes()
    assert not [path for path in local.rglob("*") if path.is_dir() and path.name.startswith(".hf-download-")]
    assert "shards/stage2_train/shard-00099.tar.gz" not in hub.downloads


PACK_NOTEBOOK = ML / "colab" / "pack_stage2_colab_cpu.ipynb"


def test_pack_stage2_colab_cpu_notebook_is_valid(tmp_path):
    from data import bimodal_pack

    notebook = json.loads(PACK_NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    cells = notebook["cells"]
    assert all(cell.get("id") for cell in cells) and len({cell["id"] for cell in cells}) == len(cells)
    header = "".join(cells[0]["source"])
    assert cells[0]["cell_type"] == "markdown"
    for needle in ("CPU", "High-RAM", "HF_TOKEN", "Secrets", "Run all", "Wosca/math-recogniser-shards"):
        assert needle in header, needle
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 5 and all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code)
    sources = ["".join(cell["source"]) for cell in code]
    for source in sources:
        ast.parse(source)
        assert all(not line.lstrip().startswith(("%", "!")) for line in source.splitlines())
    text = "\n".join(sources)
    for pattern in (r'^HF_SHARDS_REPO = "Wosca/math-recogniser-shards"$', r"^AUTO_DISCONNECT = True ",
                    r"^WORKERS = max\(2, min\(12, os\.cpu_count\(\) or 2\)\)$", r"^AST_COUNT = 30000$"):
        assert re.search(pattern, sources[0], re.M), pattern
    for needle in ('userdata.get("HF_TOKEN")', 'os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"',
                   '"pyyaml", "antlr4-python3-runtime>=4.11,<4.12", "pyarrow"', '"math-recogniser.zip"',
                   '"seed/stage2_seed.tar.gz"', "manifest_seed.json", "shards/val_dev/shard-00000.tar.gz",
                   "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz", '"-C", "-"',
                   '"ConstantHao/HME100K"', '"68ea96bd2527aac3799f94803a5f86b92fd9c2d2"',
                   '"data/train-00000-of-00002.parquet"', '"data/train-00001-of-00002.parquet"', "token=False",
                   "hashlib.sha256(data)", "--download-detexify", "verify_manifest_checksums", "compresslevel=1",
                   "upload_large_folder", 'repo_type="dataset"', "list_repo_files", "runtime.unassign()",
                   "time.sleep(15)"):
        assert needle in text, needle
    assert "install torch" not in text.lower() and "drive.mount" not in text
    assert all(not modes for source in sources for modes in mount_modes(source))

    command = next(node.value for node in ast.parse(sources[3]).body if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], "id", None) == "command" and isinstance(node.value, ast.List))
    existing = tmp_path / "tokenizer.json"
    existing.write_text("{}", encoding="utf-8")
    argv = [element.value if isinstance(element, ast.Constant)
            else "2" if getattr(element, "id", None) in ("WORKERS", "AST_COUNT") else str(existing)
            for element in command.elts]
    assert argv[1:4] == ["-u", "-m", "data.bimodal_pack"]
    args = bimodal_pack.parse_args(argv[4:])
    assert args.stages == ["stage2"] and args.fixed_tokenizer == existing and args.block_test_writers
    assert args.with_detexify and args.shard_size == 5000 and args.max_tokens == 256
    assert args.crohme_upsample == 3.0 and args.stage2_hme_fraction == 0.3


def test_colab_notebook_drive_download_retries_interstitials(tmp_path, monkeypatch):
    import urllib.request

    tree = ast.parse(notebook_cells()[2])
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "download_drive_file"
             or isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "DRIVE_DOWNLOAD_URL"]
    namespace = {"os": os, "Path": Path, "random": __import__("random"), "shutil": shutil, "time": time,
                 "urllib": urllib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "cell-2", "exec"), namespace)
    payload = gzip.compress(b"tar bytes" * 100)
    responses = [b"<!DOCTYPE html><html>Google Drive - Virus scan warning</html>", payload[:10], payload]
    requested = []

    def urlopen(request, timeout):
        requested.append(request.full_url)
        return io.BytesIO(responses.pop(0))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    destination = tmp_path / "shards" / "shard-00000.tar.gz"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        namespace["download_drive_file"]("FILE_ID_123", destination, b"\x1f\x8b", len(payload))
    assert destination.read_bytes() == payload and not list(destination.parent.glob("*.tmp"))
    assert requested == ["https://drive.usercontent.google.com/download?id=FILE_ID_123&export=download&confirm=t"] * 3
    assert "Drive interstitial" in printed.getvalue() and "expected" in printed.getvalue()
    responses[:] = [b"<html>quota exceeded</html>"] * 2
    with pytest.raises(RuntimeError, match="failed"), contextlib.redirect_stdout(io.StringIO()):
        namespace["download_drive_file"]("FILE_ID_123", tmp_path / "x.zip", b"PK\x03\x04", retries=2)
    assert not (tmp_path / "x.zip").exists()


# --------------------------------------------------------------------------- real local CUDA smoke


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.skipif(not (DATASETS / "canonical" / "crohme23" / "train.jsonl").is_file(),
                    reason="local datasets unavailable")
def test_cuda_bf16_smoke_three_stages():
    base = Path(tempfile.mkdtemp(prefix="bimodal-smoke-", dir=os.environ.get("TEMP") or None))
    timings: dict[str, float] = {}

    def timed(name: str, *command) -> str:
        started = time.perf_counter()
        result = subprocess.run([sys.executable, *map(str, command)], cwd=ML, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        timings[name] = round(time.perf_counter() - started, 1)
        logs = result.stdout + result.stderr
        assert result.returncode == 0, logs[-4000:]
        assert "non-finite" not in logs and "Traceback" not in logs, logs[-4000:]
        return logs

    try:
        shards, runs = base / "shards", base / "runs"
        timed("pack", "-m", "data.bimodal_pack", "--smoke", "--output", shards, "--workers", "4")
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        config["validation"].update(batch_size=32, max_gen_tokens=48)
        config["stage3"].update(max_gen_tokens=48)
        (base / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        shared = ["--config", base / "config.yaml", "--shards", shards, "--device", "cuda", "--num-workers", "2"]
        train = [*shared, "--no-pretrained", "--max-wall-minutes", "0", "--log-every", "2", "--compile"]
        timed("stage1", "train_bimodal.py", "--stage", 1, "--output", runs / "s1", "--batch-size", 16,
              "--max-steps", 6, *train)
        logs = timed("stage2", "train_bimodal.py", "--stage", 2, "--output", runs / "s2", "--batch-size", 16,
                     "--max-steps", 6, "--init-from", runs / "s1" / "best.pt", *train)
        assert "freshly initialized" in logs
        timed("mine", "train_bimodal.py", "--mine-hard", "--init-from", runs / "s2" / "best.pt", "--mine-limit", 24,
              "--anchor-limit", 8, *shared)
        timed("stage3", "train_bimodal.py", "--stage", 3, "--output", runs / "s3", "--batch-size", 4,
              "--max-steps", 3, "--init-from", runs / "s2" / "best.pt", *train)
        timed("eval", "train_bimodal.py", "--eval-only", "--init-from", runs / "s3" / "best.pt",
              "--output", runs / "eval", *shared)
        peaks = {}
        for name in ("s1", "s2", "s3"):
            status = read_json(runs / name / "status.json")
            assert status["state"] == "completed" and status["global_step"] > 0, status
            records = read_metrics(runs / name / "metrics.jsonl")
            for record in records:
                assert all(math.isfinite(value) for value in record["train"].values()), record
            peaks[name] = max(record.get("peak_memory_gib", 0) for record in records)
        train3 = read_metrics(runs / "s3" / "metrics.jsonl")[-1]["train"]
        assert train3["kl"] >= 0 and math.isfinite(train3["policy_loss"]) and train3["reward_std"] >= 0
        metrics = read_json(runs / "eval" / "eval_metrics.json")
        assert metrics["samples"] > 0 and metrics["by_source"]
        print("CUDA smoke timings (s):", json.dumps(timings), "peak GiB:", json.dumps(peaks))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    assert not base.exists()
