#!/usr/bin/env python3
"""CPU-friendly checks for the public source and curated model artifacts.

The default checks use only the standard library. --onnx additionally needs
numpy and onnx; it checks the graph without installing a GPU training stack.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/history2_hybrid_v1"
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".toml", ".yaml", ".yml", ".xml", ".json", ".cc", ".cpp", ".h", ".svg"}


def public_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    return sorted({ROOT / name.decode() for name in result.stdout.split(b"\0") if name})


def check_sources(paths: list[Path]) -> None:
    for path in paths:
        assert path.stat().st_size < 20 * 1024 * 1024, f"Unexpected large file: {path.relative_to(ROOT)}"
        assert not path.is_symlink(), f"Unexpected symlink: {path.relative_to(ROOT)}"
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
        elif path.suffix == ".json":
            json.loads(path.read_text())

    secrets = re.compile(
        r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
        r"|/home/(?:big|run|lenovo)/"
    )
    for path in paths:
        if path.suffix not in TEXT_SUFFIXES:
            continue
        contents = path.read_text()
        assert not secrets.search(contents), f"Potential private data: {path.relative_to(ROOT)}"
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", contents):
                if target.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                relative = unquote(target.split("#", 1)[0].strip("<>"))
                assert (path.parent / relative).exists(), f"Broken link in {path.relative_to(ROOT)}: {target}"
    print(f"Source syntax, relative links and privacy scan: {len(paths)} files OK")


def check_artifacts() -> dict:
    manifest = json.loads((MODEL / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = MODEL / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Checksum mismatch: {name}"
    video = ROOT / "docs/media/simulation.mp4"
    assert hashlib.sha256(video.read_bytes()).hexdigest() == "ddad1a4052491980d75fe33a4d4df709496eefff3279071cc7a36e1746bdf8ef", "Video checksum mismatch"
    assert (ROOT / "docs/media/simulation.gif").stat().st_size < 8 * 1024 * 1024
    hardware_video = ROOT / "docs/media/hardware.mp4"
    assert hashlib.sha256(hardware_video.read_bytes()).hexdigest() == "4845b1a794b3fd2708d18244962072baf93c9c3b349c9cb1f9f247be7450f8c2", "Hardware video checksum mismatch"
    assert (ROOT / "docs/media/hardware.gif").stat().st_size < 8 * 1024 * 1024
    for stem, digest in {
        "simulation-full-speed": "90472692f148755bbb3db3f8d23bd8d840b10540ab824a5228663dd001e976d6",
        "simulation-low-speed": "f2b389f1d7a0ee2fd565801f58b36c38dc32f15d9585607940adf05eb4aa378a",
    }.items():
        assert hashlib.sha256((ROOT / f"docs/media/{stem}.mp4").read_bytes()).hexdigest() == digest, f"Video checksum mismatch: {stem}"
        assert (ROOT / f"docs/media/{stem}.gif").stat().st_size < 8 * 1024 * 1024

    reports = sorted((ROOT / "docs/evaluations").glob("*.json"))
    assert len(reports) == 8, "Expected eight archived evaluation reports"
    stage_count, minimum_survival, maximum_tilt = 0, 1.0, 0.0
    configurations = set()
    for path in reports:
        report = json.loads(path.read_text())
        assert report["passed"] is True, f"Failed report: {path.name}"
        assert report["num_envs"] == 32
        assert (ROOT / report["checkpoint"]).is_file()
        assert (ROOT / report["zero_command_checkpoint"]).is_file()
        configurations.add((report["profile"], report["environment"], report["seed"]))
        for name, metrics in report["stages"].items():
            assert metrics["passed"] is True, f"Failed stage {path.name}:{name}"
            stage_count += 1
            minimum_survival = min(minimum_survival, metrics["survival_rate"])
            maximum_tilt = max(maximum_tilt, metrics["tilt_p95_deg"])
    assert configurations == {(p, e, s) for p in ("full", "omni-low-speed") for e in ("nominal", "robust") for s in (42, 7)}
    assert stage_count == manifest["simulation"]["stage_results"] == 120
    assert minimum_survival == 1.0
    assert abs(maximum_tilt - manifest["simulation"]["maximum_stage_tilt_p95_deg"]) < 0.001
    print(f"Artifact SHA-256 and archived metrics: 8 reports, {stage_count} stages OK")
    return manifest


def check_onnx(manifest: dict) -> None:
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator

    graph = onnx.load(MODEL / "policy.onnx", load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata["metadata_contract"] == "agibot_x2_velocity_v2"
    assert metadata["observation_history_length"] == "2"
    assert metadata["observation_dims"] == "3,3,3,2,31,31,31"
    assert metadata["observation_names"] == "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"
    assert metadata["gain_recipe"] == "factory-recorded"
    assert float(metadata["policy_dt"]) == 0.02
    assert float(metadata["phase_period"]) == 0.6
    for key in ("joint_names", "default_joint_pos", "action_scale", "joint_stiffness", "joint_damping", "joint_effort_limit", "joint_limit_lower", "joint_limit_upper"):
        values = metadata[key].split(",")
        assert len(values) == 31, f"Metadata length: {key}"
        if key == "joint_names":
            assert len(set(values)) == 31
        else:
            assert np.isfinite(np.asarray(values, dtype=np.float64)).all()
    assert metadata["hybrid_moving_checkpoint_sha256"] == manifest["files"]["model_moving_4740.pt"]
    assert metadata["hybrid_settling_checkpoint_sha256"] == manifest["files"]["model_settling_4720.pt"]
    for tensors, expected in ((graph.graph.input, manifest["input"]), (graph.graph.output, manifest["output"])):
        assert len(tensors) == 1
        tensor = tensors[0]
        assert tensor.name == expected["name"]
        assert tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert [dim.dim_value for dim in tensor.type.tensor_type.shape.dim] == expected["shape"]

    evaluator = ReferenceEvaluator(graph)
    for command in ((0, 0, 0), (0.35, 0, 0), (-0.2, 0, 0), (0, 0.15, 0), (0, -0.15, 0), (0, 0, 0.25)):
        observation = np.zeros((1, 208), dtype=np.float32)
        observation[0, [8, 11]] = -1.0
        observation[0, 12:15] = command
        observation[0, 15:18] = command
        observation[0, [19, 21]] = 1.0
        action = evaluator.run(None, {"obs": observation})[0]
        assert action.shape == (1, 31) and np.isfinite(action).all()
    print("ONNX graph, metadata and six CPU inference cases: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", action="store_true", help="Also verify the ONNX graph and CPU inference")
    args = parser.parse_args()
    check_sources(public_files())
    manifest = check_artifacts()
    if args.onnx:
        check_onnx(manifest)
    print("RELEASE_CHECK_OK")


if __name__ == "__main__":
    main()
