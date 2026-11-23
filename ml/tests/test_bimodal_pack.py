from __future__ import annotations

import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch
from torch.utils.data import DataLoader

from data import bimodal_pack as pack
from data.bimodal_pack import LeakageIndex, ink_geometry_hash
from data.bimodal_raster import (
    compute_dhash64, decode_png, encode_png, hamming64, normalize_offline_image_to_strip, render_strokes_to_strip,
)
from data.bimodal_shards import (
    SYNTHETIC_SOURCES, BimodalAugmentConfig, BimodalEvalDataset, BimodalShardDataset, augment_stroke_features,
    collate_bimodal_batch, iter_shard, sha256_file, verify_manifest_checksums,
)
from data.preprocessing import FEATURE_NAMES
from tokenizer.latex_tokenizer import LatexTokenizer

LABELS = ["x+1", "a^{2}+b^{2}", r"\frac{1}{2}", "y=2x", "z_{1}", "C+c", r"\gamma r", "8f", "u+v", "X-x",
          r"\sqrt{2}", "1,2"]


def glyph_strokes(seed: int, glyphs: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    strokes = []
    for glyph in range(glyphs):
        for _ in range(int(rng.integers(1, 3))):
            t = np.linspace(0, 1, int(rng.integers(12, 30)))
            phase, freq = rng.uniform(0, 6.28), rng.uniform(1, 3)
            x = glyph * 120 + 20 + 60 * t + 15 * np.sin(freq * 6.28 * t + phase)
            y = 100 + 60 * np.cos(freq * 3.14 * t + phase) + rng.normal(0, 1.5, len(t))
            strokes.append(np.stack([x, y], axis=1))
    return strokes


def canonical_record(sample_id: str, split: str, latex: str, strokes, *, timed: bool, writer: str | None = None):
    clock = 0.0
    out = []
    for index, stroke in enumerate(strokes):
        points = []
        for x, y in stroke:
            point = {"x": float(x), "y": float(y)}
            if timed:
                point["time"] = clock
                clock += 12.0
            points.append(point)
        clock += 150.0
        out.append({"id": index, "points": points})
    return {"schemaVersion": "stroke-latex-v1", "id": sample_id, "originalId": sample_id, "source": "external",
            "sourceDataset": "crohme", "split": split, "synthetic": False, "writerId": writer, "latex": latex,
            "latexOriginal": latex, "latexNormalized": latex, "strokes": out, "metadata": {}}


def write_inkml(path: Path, label: str, strokes, split: str, creation: str = "human") -> None:
    clock, traces = 0.0, []
    for index, stroke in enumerate(strokes):
        values = []
        for x, y in stroke:
            values.append(f"{x:.2f} {y:.2f} {clock:.1f}")
            clock += 10.0
        clock += 200.0
        traces.append(f'<trace id="{index}">{",".join(values)}</trace>')
    escaped = label.replace("&", "&amp;").replace("<", "&lt;")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<ink xmlns="http://www.w3.org/2003/InkML">'
        f'<annotation type="label">{escaped}</annotation>'
        f'<annotation type="normalizedLabel">{escaped}</annotation>'
        f'<annotation type="splitTagOriginal">{split}</annotation>'
        f'<annotation type="inkCreationMethod">{creation}</annotation>'
        f'<annotation type="sampleId">{path.stem}</annotation>'
        '<traceFormat><channel name="X" type="decimal"/><channel name="Y" type="decimal"/>'
        '<channel name="T" type="decimal"/></traceFormat>' + "".join(traces) + "</ink>",
        encoding="utf-8")


def handwriting_image(seed: int, dark_ink_on_light: bool = True) -> Image.Image:
    rng = np.random.default_rng(seed)
    background, ink = (235, 30) if dark_ink_on_light else (10, 220)
    image = Image.new("L", (int(rng.integers(220, 420)), int(rng.integers(50, 90))), background)
    draw = ImageDraw.Draw(image)
    for glyph in range(int(rng.integers(3, 7))):
        x0 = 15 + glyph * 45
        points = [(x0 + float(rng.uniform(0, 35)), float(rng.uniform(10, image.height - 10))) for _ in range(4)]
        draw.line(points, fill=ink, width=3)
    return image


# --------------------------------------------------------------------------- raster


def test_render_preserves_aspect_and_width_contract():
    wide = [np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 100.0], [0.0, 100.0], [0.0, 0.0]])]
    image, width = render_strokes_to_strip(wide)
    assert image.mode == "LA" and image.height == 128
    assert image.width % 16 == 0 and 64 <= image.width <= 640
    ink = np.asarray(image.getchannel("L"))
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    ratio = (cols[-1] - cols[0]) / (rows[-1] - rows[0])
    assert 3.6 < ratio < 4.4
    assert width <= image.width and width >= cols[-1]
    assert ink[0, 0] == 0 and ink[-1, -1] == 0 and ink.max() == 255

    very_wide = [np.array([[0.0, 0.0], [10000.0, 50.0]])]
    image, width = render_strokes_to_strip(very_wide)
    assert image.width == 640 and width <= 640


def test_render_order_channel_is_monotonic_and_masked():
    line = np.stack([np.linspace(0, 300, 120), np.full(120, 0.0)], axis=1)
    descender = np.stack([np.full(30, 150.0), np.linspace(-60, 60, 30)], axis=1)
    image, _ = render_strokes_to_strip([line])
    ink, order = np.asarray(image.getchannel("L")), np.asarray(image.getchannel("A"))
    assert np.array_equal(order > 0, ink > 0) and order[ink == 0].max(initial=0) == 0
    columns = np.flatnonzero(ink.any(axis=0))
    peaks = order[:, columns].max(axis=0).astype(int)
    assert np.all(np.diff(peaks) >= 0) and peaks[0] <= 3 and peaks[-1] == 255

    image, _ = render_strokes_to_strip([line, descender])
    ink, order = np.asarray(image.getchannel("L")), np.asarray(image.getchannel("A"))
    assert np.array_equal(order > 0, ink > 0)
    first_stroke = order[ink.shape[0] // 2, : image.width // 4]
    second_stroke = order[5:20, :].max()
    assert first_stroke[first_stroke > 0].max() < second_stroke
    assert order[order > 0].min() >= 1 and order.max() == 255


@pytest.mark.parametrize("strokes", [
    [np.array([[5.0, 5.0]])],
    [np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])],
    [np.array([[0.0, 0.0], [0.0, 80.0]])],
    [np.array([[0.0, 3.0], [90.0, 3.0]])],
    [np.array([[1e6, -1e6], [1e6 + 3, -1e6 + 5]]), np.array([[np.nan, 1.0], [1e6 + 1, -1e6 + 2]])],
    [np.zeros((0, 2)), np.array([[1.0, 2.0, 99.0]])],
])
def test_render_degenerate_strokes_are_safe(strokes):
    image, width = render_strokes_to_strip(strokes)
    ink, order = np.asarray(image.getchannel("L")), np.asarray(image.getchannel("A"))
    assert image.height == 128 and image.width % 16 == 0 and 16 <= width <= image.width
    assert ink.max() > 0 and np.array_equal(order > 0, ink > 0)


def test_render_empty_input_is_blank():
    image, width = render_strokes_to_strip([])
    assert image.size == (64, 128) and np.asarray(image).max() == 0 and 16 <= width <= 64


# --------------------------------------------------------------------------- offline normalization


def _ink_profile(image: Image.Image) -> np.ndarray:
    ink = np.asarray(image.getchannel("L"), dtype=np.float32)
    return ink.sum(axis=0)[: image.width]


def test_offline_inverts_dark_ink_and_preserves_inverted_input():
    light = handwriting_image(3, dark_ink_on_light=True)
    inverted = Image.eval(light, lambda v: 255 - v)
    out_light, width_light = normalize_offline_image_to_strip(light)
    out_dark, width_dark = normalize_offline_image_to_strip(inverted)
    for image, width in ((out_light, width_light), (out_dark, width_dark)):
        ink = np.asarray(image.getchannel("L"))
        assert image.mode == "LA" and image.height == 128 and image.width % 16 == 0
        assert 64 <= image.width <= 640 and width <= image.width
        assert ink[:, 0].max() == 0 and ink[0].max() == 0 and ink[:, width:].max(initial=0) == 0
        assert ink.max() == 255 and (ink > 0).mean() < 0.5
        assert np.asarray(image.getchannel("A")).max() == 0
    assert out_light.size == out_dark.size
    a, b = _ink_profile(out_light), _ink_profile(out_dark)
    assert np.corrcoef(a, b)[0, 1] > 0.95


def test_offline_handles_dim_photo_bytes_and_paths(tmp_path):
    rng = np.random.default_rng(0)
    paper = np.clip(rng.normal(95, 4, (60, 300)), 0, 255)
    image = Image.fromarray(paper.astype(np.uint8), "L")
    draw = ImageDraw.Draw(image)
    draw.line([(20, 15), (60, 45), (100, 15), (150, 45)], fill=50, width=3)
    draw.ellipse((200, 40, 205, 45), fill=45)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    out, _ = normalize_offline_image_to_strip(buffer.getvalue())
    ink = np.asarray(out.getchannel("L"))
    assert ink[:, :4].max() == 0 and ink.max() == 255 and (ink > 128).mean() < 0.3
    path = tmp_path / "sample.img"
    path.write_bytes(buffer.getvalue())
    out_path, _ = normalize_offline_image_to_strip(path)
    assert np.array_equal(np.asarray(out_path), np.asarray(out))
    blank, width = normalize_offline_image_to_strip(Image.new("L", (100, 40), 255))
    assert np.asarray(blank).max() == 0 and blank.size == (64, 128) and width == 64


def test_dhash_is_stable_and_discriminative():
    image, _ = render_strokes_to_strip(glyph_strokes(1))
    same, _ = render_strokes_to_strip([stroke * 3 + 7 for stroke in glyph_strokes(1)])
    other, _ = render_strokes_to_strip(glyph_strokes(2))
    assert len(compute_dhash64(image)) == 16
    assert hamming64(compute_dhash64(image), compute_dhash64(same)) <= 2
    assert hamming64(compute_dhash64(image), compute_dhash64(other)) > 8
    png = encode_png(image)
    restored = decode_png(png)
    assert restored.shape == (2, 128, image.width) and np.array_equal(restored[0], np.asarray(image.getchannel("L")))


# --------------------------------------------------------------------------- leakage auditor


def _test_index() -> tuple[LeakageIndex, list[np.ndarray], str]:
    index = LeakageIndex(reserved_writers={"held_out_writer"})
    strokes = glyph_strokes(10)
    image, _ = render_strokes_to_strip(strokes)
    dhash = compute_dhash64(image)
    index.add(test_id="crohme:test/CROHME2023_test/0", source="crohme_test", normalized_latex="x+1",
              dhash=dhash, ink_hash=ink_geometry_hash(strokes))
    return index, strokes, dhash


def test_leakage_catches_latex_and_ink_hash_matches():
    index, strokes, dhash = _test_index()
    moved = [stroke * 2.5 + np.array([300.0, -40.0]) for stroke in strokes]
    moved_dhash = compute_dhash64(render_strokes_to_strip(moved)[0])
    hit = index.check(sample_id="crohme:train/a", normalized_latex="x+1", dhash=moved_dhash,
                      ink_hash=ink_geometry_hash(moved))
    assert hit == ("latex_ink_hash", "crohme:test/CROHME2023_test/0")
    relabelled = index.check(sample_id="crohme:train/b", normalized_latex="x+7", dhash=moved_dhash,
                             ink_hash=ink_geometry_hash(moved))
    assert relabelled[0] == "ink_hash"
    jittered = [stroke + np.random.default_rng(0).normal(0, 0.3, stroke.shape) for stroke in strokes]
    near = index.check(sample_id="crohme:train/c", normalized_latex="x+1", dhash=pack.audit_dhash(jittered),
                       ink_hash=ink_geometry_hash(jittered))
    assert near is not None and near[0] in ("latex_dhash", "latex_ink_hash")
    assert index.check(sample_id="crohme:test/CROHME2023_test/0", normalized_latex="q", dhash=None)[0] == "test_id"
    assert index.check(sample_id="w", normalized_latex="q", dhash=None, writer_id="held_out_writer")[0] == \
        "reserved_writer"


def test_leakage_catches_exact_and_near_dhash_offline():
    index, _, dhash = _test_index()
    value = int(dhash, 16)
    assert index.check(sample_id="hme:1", normalized_latex="other", dhash=dhash)[0] == "dhash_exact"
    one_bit = f"{value ^ (1 << 5):016x}"
    two_bits = f"{value ^ (1 << 5) ^ (1 << 40):016x}"
    three_bits = f"{value ^ 0b111:016x}"
    assert index.check(sample_id="hme:2", normalized_latex="other", dhash=one_bit)[0] == "dhash_near"
    assert index.check(sample_id="hme:3", normalized_latex="other", dhash=two_bits)[0] == "dhash_near"
    assert index.check(sample_id="hme:4", normalized_latex="other", dhash=three_bits) is None
    restored = LeakageIndex.from_dict(json.loads(json.dumps(index.to_dict())))
    assert restored.check(sample_id="hme:2", normalized_latex="other", dhash=one_bit)[0] == "dhash_near"
    assert restored.reserved_writers == {"held_out_writer"}


def test_leakage_allows_clean_samples_and_guards_blank_hashes():
    index, _, _ = _test_index()
    clean = glyph_strokes(99)
    clean_dhash = compute_dhash64(render_strokes_to_strip(clean)[0])
    assert index.check(sample_id="crohme:train/z", normalized_latex="x+1", dhash=clean_dhash,
                       ink_hash=ink_geometry_hash(clean)) is None
    offline = compute_dhash64(normalize_offline_image_to_strip(handwriting_image(5))[0])
    assert index.check(sample_id="hme:9", normalized_latex="y", dhash=offline) is None
    index.add(test_id="t:blank", source="crohme_test", normalized_latex="-", dhash="0000000000000000")
    assert index.check(sample_id="hme:blank", normalized_latex="1", dhash="0000000000000001") is None
    assert index.check(sample_id="hme:blank", normalized_latex="-", dhash="0000000000000001")[0] == "dhash_near"


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("bimodal_corpus")
    crohme, mathwriting, hme = root / "crohme", root / "mathwriting", root / "hme"
    crohme.mkdir()
    test_ink = {i: glyph_strokes(1000 + i) for i in range(6)}
    tests = [canonical_record(f"crohme:test/CROHME2019_test/{i}" if i < 3 else f"crohme:test/CROHME2023_test/{i}",
                              "test", LABELS[i], test_ink[i], timed=False, writer=f"test_writer_{i}")
             for i in range(6)]
    train = []
    for i in range(24):
        strokes, label, writer = glyph_strokes(i), LABELS[i % len(LABELS)], f"writer_{i % 5}"
        if i == 0:
            strokes, label = [s * 1.7 + 25 for s in test_ink[0]], LABELS[0]
        if i == 1:
            writer = "RESERVED_W"
        train.append(canonical_record(f"crohme:train/T/{i}", "train", label, strokes, timed=i % 2 == 0,
                                      writer=writer))
    val = [canonical_record(f"crohme:val/V/{i}", "validation", LABELS[i % len(LABELS)], glyph_strokes(500 + i),
                            timed=False) for i in range(8)]
    for name, records in (("train", train), ("validation", val), ("test", tests)):
        with (crohme / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    mw_test = glyph_strokes(2000)
    for i in range(4):
        write_inkml(mathwriting / "test" / f"t{i:03d}.inkml", LABELS[i], mw_test if i == 0 else glyph_strokes(2000 + i),
                    "test")
    for i in range(16):
        write_inkml(mathwriting / "train" / f"m{i:03d}.inkml", LABELS[(i + 3) % len(LABELS)],
                    mw_test if i == 0 else glyph_strokes(3000 + i), "train")
    for i in range(8):
        write_inkml(mathwriting / "valid" / f"v{i:03d}.inkml", LABELS[i], glyph_strokes(4000 + i), "valid")
        write_inkml(mathwriting / "synthetic" / f"s{i:03d}.inkml", LABELS[i], glyph_strokes(5000 + i),
                    "synthetic", creation="synthetic")
    for i, label in enumerate(["+", "+", "z", "C", "c", "y", r"\gamma", "r"]):
        write_inkml(mathwriting / "symbols" / f"sym{i:03d}.inkml", label, glyph_strokes(6000 + i, glyphs=1),
                    "symbols")

    (hme / "images").mkdir(parents=True)
    rows = []
    leaked_view = render_strokes_to_strip(test_ink[1])[0].getchannel("L")
    for i in range(14):
        image = Image.eval(leaked_view, lambda v: 255 - v) if i == 0 else handwriting_image(7000 + i)
        image.save(hme / "images" / f"h{i}.img", format="PNG")
        rows.append({"id": f"hme100k:{i}", "image": f"images/h{i}.img", "latex": LABELS[i % len(LABELS)],
                     "split": "validation" if i >= 12 else "train", "writer": None})
    (hme / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    structure = {"format_version": 1, "split": "train", "symbols": ["x", "+", "1", "y"],
                 "relation_classes": ["None", "Right", "Sup"], "records": {}}
    for i in (2, 3, 4):
        count = len(train[i]["strokes"])
        structure["records"][f"crohme:train/T/{i}"] = {
            "stroke_groups": [min(j, 2) for j in range(count)], "symbol_classes": [0, 1, 2][: min(count, 3)],
            "relations": [[0, 1, 1], [1, 2, 1]][: max(0, min(count, 3) - 1)]}
    structure["records"]["crohme:train/T/5"] = {"stroke_groups": [0], "symbol_classes": [3], "relations": []}
    (root / "structure.json").write_text(json.dumps(structure), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def packed(corpus):
    output = corpus / "shards"
    code = pack.main([
        "--output", str(output), "--workers", "0", "--shard-size", "8", "--seed", "7",
        "--crohme-root", str(corpus / "crohme"), "--mathwriting-root", str(corpus / "mathwriting"),
        "--hme-root", str(corpus / "hme"), "--structure", str(corpus / "structure.json"),
        "--external-root", str(corpus / "external"), "--reserved-writers", "RESERVED_W",
        "--stage2-hme-fraction", "1.0", "--ast-count", "6",
    ])
    assert code == 0
    return output


def test_manifest_checksums_and_audit(packed):
    manifest = json.loads((packed / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["stages"]) == {"stage1", "stage2", "val"}
    assert manifest["leakage"]["complete"] and manifest["leakage"]["test_samples"] == 10
    for stage in manifest["stages"].values():
        assert stage["shards"]
        for shard in stage["shards"]:
            assert sha256_file(packed / stage["directory"] / shard["file"]) == shard["sha256"]
    assert all(verify_manifest_checksums(packed).values())
    assert manifest["preprocessing"]["normalization"] == "glyph" and manifest["preprocessing"]["max_points"] == 2048
    assert manifest["structure"]["relation_classes"] == ["None", "Right", "Sup"]

    report = json.loads((packed / "audit_report.json").read_text(encoding="utf-8"))
    leaks = {(leak["stage"], leak["id"]): leak["reason"] for leak in report["leaks"]}
    for stage in ("stage1", "stage2"):
        assert leaks[(stage, "crohme:train/T/0")] == "latex_ink_hash"
        assert leaks[(stage, "crohme:train/T/1")] == "reserved_writer"
        assert leaks[(stage, "mathwriting:m000")] in ("ink_hash", "latex_ink_hash")
        assert leaks[(stage, "hme100k:0")] in ("dhash_exact", "dhash_near")
    stage1 = manifest["stages"]["stage1"]
    assert stage1["dropped_leakage"] == {"crohme_train": 2, "mathwriting_train": 1, "hme100k": 1}
    assert stage1["sources"] == {"crohme_train": 22, "mathwriting_train": 15, "hme100k": 11}
    assert manifest["stages"]["val"]["sources"] == {"crohme_val": 8, "mathwriting_val": 8}
    assert manifest["stages"]["stage2"]["with_structure"] == 3

    test_ids = {json.loads(line)["id"] for line in (packed.parent / "crohme" / "test.jsonl").open()}
    for stage in manifest["stages"].values():
        for shard in stage["shards"]:
            for record in iter_shard(packed / stage["directory"] / shard["file"]):
                meta = json.loads(record["json"])
                assert meta["id"] not in test_ids and meta["writer_id"] != "RESERVED_W"
                assert meta["split"] != "test" and meta["id"] != "hme100k:12"


def test_shard_sample_contract(packed):
    keys = {"id", "source", "split", "latex", "normalized_latex", "has_strokes", "has_timing", "image_width",
            "stroke_count", "writer_id"}
    seen_structure = 0
    for directory in ("stage1_train", "stage2_train", "val_dev"):
        for path in sorted((packed / directory).glob("shard-*.tar")):
            for record in iter_shard(path):
                meta = json.loads(record["json"])
                assert keys <= set(meta)
                image = Image.open(io.BytesIO(record["png"]))
                assert image.mode == "LA" and image.height == 128 and image.width % 16 == 0
                assert 64 <= image.width <= 640 and 16 <= meta["image_width"] <= image.width
                ink, order = np.asarray(image.getchannel("L")), np.asarray(image.getchannel("A"))
                assert order[ink == 0].max(initial=0) == 0
                if meta["source"] == "hme100k":
                    assert order.max() == 0 and not meta["has_strokes"]
                else:
                    assert order[ink > 0].min() >= 1
                assert (record["npz"] is not None) == meta["has_strokes"]
                if directory != "stage1_train" and meta["source"] != "hme100k":
                    assert meta["has_strokes"]
                if directory == "stage1_train":
                    assert not meta["has_strokes"]
                    if meta["source"] == "crohme_train":
                        assert meta["upsample_hint"] == 3.0
                if record["npz"] is not None:
                    arrays = np.load(io.BytesIO(record["npz"]))
                    features = arrays["features"]
                    assert features.dtype == np.float16 and features.shape[1] == len(FEATURE_NAMES) == 10
                    assert (features[:, 4] > 0).any() == meta["has_timing"]
                    if "stroke_symbol_ids" in arrays:
                        seen_structure += 1
                        assert arrays["stroke_symbol_ids"].dtype == np.int16
                        assert len(arrays["stroke_group_ids"]) == meta["stroke_count"]
                        assert arrays["relation_edges"].dtype == np.int16 and arrays["relation_edges"].shape[1] == 3
    assert seen_structure == 3


def test_stage1_vision_only_batches(packed):
    dataset = BimodalShardDataset(packed, stage="stage1", seed=3, shuffle_buffer=16)
    batches = list(DataLoader(dataset, batch_size=8, collate_fn=collate_bimodal_batch))
    total = sum(len(batch["ids"]) for batch in batches)
    assert total == 48
    for batch in batches:
        images = batch["images"]
        assert images.dtype == torch.float32 and images.shape[1:3] == (2, 128) and images.shape[3] % 16 == 0
        assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0
        assert batch["stroke_features"] is None and batch["stroke_padding_mask"] is None
        assert not batch["has_strokes"].any()
        assert (batch["image_widths"] <= images.shape[3]).all()
        for row, source in enumerate(batch["sources"]):
            if source == "hme100k":
                assert float(images[row, 1].max()) == 0.0


def test_stage2_mixed_batches_and_r2l(packed):
    dataset = BimodalShardDataset(packed, stage="stage2", seed=5, shuffle_buffer=8)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_bimodal_batch)
    batch = next(iter(loader))
    has_strokes = batch["has_strokes"]
    assert has_strokes.any() and not has_strokes.all()
    assert set(batch["sources"]) >= {"crohme_train", "mathwriting_train", "mathwriting_synthetic", "hme100k",
                                     "ast_synthetic"}
    features, mask = batch["stroke_features"], batch["stroke_padding_mask"]
    assert features.shape[:2] == mask.shape and features.shape[2] == 10
    assert mask[~has_strokes].all() and (~mask[has_strokes]).any(dim=1).all()
    assert (features[mask] == 0).all()
    tokenizer = dataset.tokenizer
    for row in range(len(batch["ids"])):
        tokens = batch["tokens"][row][~batch["token_padding_mask"][row]].tolist()
        r2l = batch["r2l_tokens"][row][~batch["r2l_token_padding_mask"][row]].tolist()
        assert tokens[0] == r2l[0] == tokenizer.bos_id and tokens[-1] == r2l[-1] == tokenizer.eos_id
        assert r2l[1:-1] == tokens[-2:0:-1]
        assert tokenizer.decode(tokens) == tokenizer.decode(tokenizer.encode(batch["normalized_latex"][row]))
        if batch["sources"][row] == "hme100k":
            assert not has_strokes[row]
    structured = [target for target in batch["structure_targets"] if target is not None]
    assert len(structured) == 3 and all(target["relation_edges"].shape[1] == 3 for target in structured)
    assert batch["tokens"].shape == batch["r2l_tokens"].shape


def test_clean_human_only_filters_synthetic(packed):
    dataset = BimodalShardDataset(packed, stage="stage2", clean_human_only=True)
    samples = list(dataset)
    sources = {sample["source"] for sample in samples}
    assert sources and not sources & SYNTHETIC_SOURCES
    assert {"crohme_train", "mathwriting_train", "hme100k"} <= sources
    everything = {sample["source"] for sample in BimodalShardDataset(packed, stage="stage2")}
    assert {"mathwriting_synthetic", "ast_synthetic"} <= everything


def test_timing_dropout_statistics(packed):
    rng = np.random.default_rng(0)
    features = np.zeros((40, 10), dtype=np.float32)
    features[:, 4] = 0.25
    features[:, :2] = rng.normal(size=(40, 2))
    config = BimodalAugmentConfig()
    dropped, scales = 0, []
    for _ in range(4000):
        out = augment_stroke_features(features, True, rng, config)
        if not out[:, 4].any():
            dropped += 1
        else:
            scales.append(float(out[0, 4] / 0.25))
    assert 0.46 < dropped / 4000 < 0.54
    assert 0.7 - 1e-6 <= min(scales) and max(scales) <= 1.3 + 1e-6 and np.std(scales) > 0.1
    untimed = features.copy()
    untimed[:, 4] = 0.0
    out = augment_stroke_features(untimed, False, rng, config)
    assert not out[:, 4].any()
    ratio = out[:, :2] / features[:, :2]
    assert 0.95 - 1e-5 <= ratio.min() and ratio.max() <= 1.05 + 1e-5

    dataset = BimodalShardDataset(packed, stage="stage2", seed=11)
    timed = zeroed = 0
    for epoch in range(6):
        dataset.set_epoch(epoch)
        for sample in dataset:
            if sample["has_timing"]:
                timed += 1
                zeroed += int(not sample["stroke_features"][:, 4].any())
    assert timed >= 100 and 0.3 < zeroed / timed < 0.7

    evaluation = BimodalShardDataset(packed, stage="stage2", train=False)
    timed_eval = [sample for sample in evaluation if sample["has_timing"]]
    assert timed_eval and all(sample["stroke_features"][:, 4].any() for sample in timed_eval)


def test_epoch_shuffle_is_deterministic(packed):
    dataset = BimodalShardDataset(packed, stage="stage1", seed=9)
    dataset.set_epoch(1)
    first = [sample["id"] for sample in dataset]
    second = [sample["id"] for sample in dataset]
    dataset.set_epoch(2)
    third = [sample["id"] for sample in dataset]
    assert first == second and first != third and sorted(first) == sorted(third)


@pytest.mark.parametrize("stage", ["stage1", "val"])
def test_multiworker_loading_has_no_duplicates_or_drops(packed, stage):
    dataset = BimodalShardDataset(packed, stage=stage, seed=1, shuffle_buffer=4)
    expected = sorted(sample["id"] for sample in BimodalShardDataset(packed, stage=stage, train=False))
    loader = DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=collate_bimodal_batch)
    ids = [identifier for batch in loader for identifier in batch["ids"]]
    assert sorted(ids) == expected
    if stage == "val":
        assert len(dataset.shards) == 1 and len(expected) == 16


def test_eval_dataset_is_deterministic(packed):
    dataset = BimodalEvalDataset(packed)
    assert len(dataset) == 16
    first, again = dataset[3], dataset[3]
    assert torch.equal(first["image"], again["image"]) and torch.equal(first["tokens"], again["tokens"])
    assert first["stroke_features"] is not None
    batch = collate_bimodal_batch([dataset[i] for i in range(len(dataset))])
    assert batch["has_strokes"].all() and batch["images"].shape[0] == 16
    assert {source for source in batch["sources"]} == {"crohme_val", "mathwriting_val"}


def test_resume_skips_matching_shards(packed, corpus, capsys):
    before = {path.name: path.stat().st_mtime_ns for path in (packed / "stage1_train").glob("shard-*.tar")}
    code = pack.main([
        "--output", str(packed), "--workers", "0", "--shard-size", "8", "--seed", "7", "--stages", "stage1",
        "--crohme-root", str(corpus / "crohme"), "--mathwriting-root", str(corpus / "mathwriting"),
        "--hme-root", str(corpus / "hme"), "--structure", str(corpus / "structure.json"),
        "--external-root", str(corpus / "external"), "--reserved-writers", "RESERVED_W",
    ])
    assert code == 0 and "already packed" in capsys.readouterr().out
    after = {path.name: path.stat().st_mtime_ns for path in (packed / "stage1_train").glob("shard-*.tar")}
    assert before == after
    manifest = json.loads((packed / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["stages"]) == {"stage1", "stage2", "val"}


def test_stage2_only_repack_keeps_fixed_tokenizer_and_seed_stages(packed, corpus, tmp_path):
    original = json.loads((packed / "manifest.json").read_text(encoding="utf-8"))
    output = tmp_path / "colab"
    (output / "val_dev").mkdir(parents=True)
    for shard in original["stages"]["val"]["shards"]:
        shutil.copyfile(packed / "val_dev" / shard["file"], output / "val_dev" / shard["file"])
    seed = {**original, "stages": {stage: original["stages"][stage] for stage in ("stage1", "val")}}
    (output / "manifest.json").write_text(json.dumps(seed), encoding="utf-8")
    (output / "tokenizer.json").write_text("stale", encoding="utf-8")
    locked = tmp_path / "seed" / "tokenizer.json"
    locked.parent.mkdir()
    tokenizer = LatexTokenizer.load(packed / "tokenizer.json")
    tokenizer.add_tokens([r"\stageonelocked"])
    tokenizer.save(locked)
    common = ["--workers", "0", "--shard-size", "8", "--seed", "7",
              "--crohme-root", str(corpus / "crohme"), "--mathwriting-root", str(corpus / "mathwriting"),
              "--hme-root", str(corpus / "hme"), "--structure", str(corpus / "structure.json"),
              "--external-root", str(corpus / "external"), "--reserved-writers", "RESERVED_W",
              "--stage2-hme-fraction", "1.0", "--ast-count", "6"]
    with pytest.raises(SystemExit):
        pack.parse_args(["--output", str(output), "--fixed-tokenizer", str(tmp_path / "missing.json")])
    code = pack.main(["--output", str(output), "--stages", "stage2", "--fixed-tokenizer", str(locked), *common])
    assert code == 0

    assert (output / "tokenizer.json").read_bytes() == locked.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["stages"]) == {"stage1", "stage2", "val"}
    assert manifest["stages"]["val"] == original["stages"]["val"]
    assert manifest["stages"]["stage1"] == original["stages"]["stage1"]
    assert manifest["stages"]["stage2"]["sources"] == original["stages"]["stage2"]["sources"]
    info = manifest["tokenizer"]
    assert info["fixed_tokenizer"] is True and info["tokenizer_sha256"] == sha256_file(locked)
    assert info["size"] == len(tokenizer) and info["oov_labels"] == 0
    assert original["tokenizer"]["fixed_tokenizer"] is False
    checks = verify_manifest_checksums(output)
    assert all(ok for key, ok in checks.items() if key.split("/")[0] in ("stage2_train", "val_dev"))
    assert not any(ok for key, ok in checks.items() if key.startswith("stage1_train/"))
    report = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
    assert set(report["stages"]) == {"stage1", "stage2", "val"}
    assert report["leak_counts"]["stage2"]


def test_smoke_mode_caps_sample_counts(corpus, tmp_path):
    code = pack.main([
        "--smoke", "--output", str(tmp_path / "smoke"), "--workers", "0",
        "--crohme-root", str(corpus / "crohme"), "--mathwriting-root", str(corpus / "mathwriting"),
        "--hme-root", str(corpus / "hme"), "--structure", str(corpus / "structure.json"),
        "--external-root", str(corpus / "external"),
    ])
    assert code == 0
    manifest = json.loads((tmp_path / "smoke" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["smoke"] and not manifest["leakage"]["complete"]
    assert manifest["stages"]["stage1"]["samples"] <= 64 and manifest["stages"]["stage2"]["samples"] <= 64
    assert manifest["stages"]["val"]["samples"] <= 32


def test_detexify_parser(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "symbols.json").write_text(json.dumps([
        {"command": r"\alpha", "mathmode": True, "id": "latex2e-OT1-_alpha"},
        {"command": r"\textless", "mathmode": False, "id": "latex2e-OT1-_textless"},
    ]), encoding="utf-8")
    import gzip
    rows = ["1\tlatex2e-OT1-_alpha\t[[[1,2,100],[3,4,120]],[[5,6,300]]]",
            "2\tlatex2e-OT1-_textless\t[[[1,1,1],[2,2,2]]]", "3\tlatex2e-OT1-_alpha\t\\N"]
    dump = "SET x;\nCOPY samples (id, key, strokes) FROM stdin;\n" + "\n".join(rows) + "\n\\.\n"
    with gzip.open(raw / "detexify.sql.gz", "wt", encoding="utf-8") as handle:
        handle.write(dump)
    summary = pack.parse_detexify(raw, tmp_path / "samples.jsonl")
    assert summary["accepted"] == 1 and summary["skipped"] == {"not_math_symbol": 1, "malformed": 1}
    record = json.loads((tmp_path / "samples.jsonl").read_text(encoding="utf-8"))
    assert record["latex"] == r"\alpha" and len(record["strokes"]) == 2
    expression = pack.detexify_expression(record)
    assert expression.stroke_count == 2 and expression.metadata["time_available"]


def test_unimer_filters():
    assert pack.unimer_label_reason(r"a \\ b", 256)[1] == "multi_line"
    assert pack.unimer_label_reason("", 256)[1] == "empty_label"
    assert pack.unimer_label_reason(r"\operatorname* { P r } ( x )", 256)[1] == "unsupported_command"
    assert pack.unimer_label_reason(" ".join(["x"] * 300), 256)[1] == "too_long"
    assert pack.unimer_label_reason(r"\frac { a } { b }", 256) == (r"\frac{a}{b}", None)
    assert pack.unimer_image_reason(100, 300) == "too_tall"
    assert pack.unimer_image_reason(1300, 100) == "too_wide"
    assert pack.unimer_image_reason(300, 60) is None


def test_unimer_squished_and_donor_cache(tmp_path):
    # Ultra-wide 20:1 UniMER strip shrinks to ~31 px tall -> rejected by build_offline.
    wide = Image.new("L", (1000, 50), 255)
    ImageDraw.Draw(wide).line([(10, 25), (990, 25)], fill=0, width=8)
    wide_path = tmp_path / "wide.png"
    wide.save(wide_path)
    normal = Image.new("L", (240, 80), 255)
    ImageDraw.Draw(normal).rectangle([20, 15, 220, 65], outline=0, width=5)
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)

    pack._init_worker(pack.PackContext(index=pack.LeakageIndex(), max_tokens=256,
                                       unimer_max_tokens=96, unimer_min_ink_height=44))
    dropped = pack.build_offline(pack.Candidate("unimer", "offline", str(wide_path), -1, "u:wide", "x+y", "train"))
    assert isinstance(dropped, pack.Drop) and dropped.reason == "squished_unimer"
    kept = pack.build_offline(pack.Candidate("unimer", "offline", str(normal_path), -1, "u:ok", "x+y", "train"))
    assert isinstance(kept, pack.Packed)
    too_long = pack.build_offline(pack.Candidate("unimer", "offline", str(normal_path), -1, "u:long",
                                                 "+".join(["x"] * 60), "train"))
    assert isinstance(too_long, pack.Drop) and too_long.reason == "too_long"

    # DonorBase cache must key by integer index, not LeakageIndex.
    ink_path = tmp_path / "donor.jsonl"
    record = canonical_record("d:0", "train", "x", glyph_strokes(1, glyphs=1), timed=True, writer="w1")
    ink_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    donor = pack.Candidate("crohme_train", "canonical", str(ink_path), 0, "d:0", "x", "train", "w1")
    base = pack.DonorBase([donor], max_tokens=256)
    first = base[0]
    assert 0 in base.cache and base[0] is first


REAL_CROHME = Path(r"E:\math-recogniser-datasets\canonical\crohme23\train.jsonl")
REAL_HME = Path(r"E:\math-recogniser-datasets\offline\hme100k\examples.jsonl")


@pytest.mark.skipif(not (REAL_CROHME.is_file() and REAL_HME.is_file()), reason="local datasets unavailable")
def test_real_samples_read_only():
    from data.canonical import expression_from_record
    with REAL_CROHME.open("rb") as handle:
        expression = expression_from_record(json.loads(handle.readline()))
    strokes = pack.expression_strokes(expression)
    image, width = render_strokes_to_strip(strokes)
    ink, order = np.asarray(image.getchannel("L")), np.asarray(image.getchannel("A"))
    assert image.height == 128 and image.width % 16 == 0 and ink.max() == 255
    assert np.array_equal(order > 0, ink > 0) and width <= image.width
    with REAL_HME.open("r", encoding="utf-8") as handle:
        row = json.loads(handle.readline())
    offline, width = normalize_offline_image_to_strip(REAL_HME.parent / row["image"])
    ink = np.asarray(offline.getchannel("L"))
    assert offline.height == 128 and ink[:, 0].max() == 0 and ink.max() == 255
    assert np.asarray(offline.getchannel("A")).max() == 0
    assert (ink > 0).mean() < 0.5
