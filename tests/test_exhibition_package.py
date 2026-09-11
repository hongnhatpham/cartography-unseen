"""Build tiny package fixtures without using production runtime, GPU or archives."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/pack_exhibition.ps1"
POWERSHELL = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(not POWERSHELL, reason="Requires Windows PowerShell")


def array_values(source: str, name: str) -> list[str]:
    match = re.search(rf"\${name} = @\((.*?)\n\)", source, re.S)
    assert match, name
    return re.findall(r"'([^']+)'", match.group(1))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    text = SCRIPT.read_text(encoding="utf-8")
    files = array_values(text, "requiredFiles")
    files += [f"tools/{name}" for name in array_values(text, "toolFiles")]
    files += ["app/main.py", "app/monitoring.py", "assets/viewer.html", "shaders/test.glsl"]
    for name in files:
        path = project / name.replace("\\", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    (project / "config.json").write_text(json.dumps({"model_path": "models/sd_turbo", "taesd_path": "models/taesd"}))
    shutil.copy2(SCRIPT, project / "tools/pack_exhibition.ps1")
    return project


def pack(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(source / "tools/pack_exhibition.ps1"), *args],
        text=True, capture_output=True, timeout=30,
    )


def seed(source: Path, names: list[str]) -> None:
    for name in names:
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("private fixture", encoding="utf-8")


def test_lightweight_manifest_and_private_data_exclusion(source: Path) -> None:
    seed(source, ["cache/INSTALL_COMPLETE.txt", "cache/journey-credentials.ini",
                  "journeys/private/manifest.json", "logs/private.log", "screenshot/private.png",
                  "dashboard/credentials/machine.json", ".git/config", "runtime/python/python.exe",
                  "models/sd_turbo/private.bin", "app/.env", "app/__pycache__/main.pyc",
                  "assets/.ssh/id_ed25519", "assets/private.key"])
    result = pack(source, "-SkipVerify")
    assert result.returncode == 0, result.stdout + result.stderr
    target = source / "dist/RealtimeDiffusionArt"
    included = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
    required = {"app/monitoring.py", "tools/setup_exhibition_windows.ps1",
                "tools/run_exhibition.ps1", "tools/exhibition_status.ps1", "tools/monitor_agent.py",
                "requirements-monitor.txt", "requirements-storage.txt", "tools/sync_journeys.py",
                "tools/configure_journey_sync.py", "tools/restore_journey.py",
                "tools/export_journey_svg.py", "docs/journey-storage.md",
                "docs/exhibition-windows.md", "dashboard/README.md"}
    assert required <= included
    assert not any(name.startswith(("cache/", "journeys/", "logs/", "screenshot/", "runtime/", "models/", ".git/", "dashboard/credentials/")) for name in included)
    assert not included & {"app/.env", "app/__pycache__/main.pyc", "assets/.ssh/id_ed25519", "assets/private.key"}
    assert "first launch requires Internet" in result.stdout


def test_prepared_cannot_skip_verification_or_destroy_existing_package(source: Path) -> None:
    seed(source, ["dist/RealtimeDiffusionArt/keep.txt"])
    result = pack(source, "-PreparedOffline", "-SkipVerify")
    assert result.returncode != 0
    assert "requires real offline verification" in result.stderr
    assert (source / "dist/RealtimeDiffusionArt/keep.txt").is_file()


def test_prepared_filters_runtime_and_never_copies_source_marker_on_failure(source: Path) -> None:
    seed(source, ["runtime/python/python.exe", "runtime/python/Lib/module.py",
                  "runtime/python/Lib/site-packages/certifi/cacert.pem",
                  "runtime/python/credentials.json", "runtime/python/.ssh/id_rsa",
                  "runtime/python/cache/token.json", "runtime/python/private.key",
                  "runtime/python/__pycache__/test.pyc", "runtime/unrelated.txt",
                  "cache/INSTALL_COMPLETE.txt"])
    pem = source / "runtime/python/secret.pem"
    pem.write_text("-----BEGIN RSA PRIVATE KEY-----\nfixture\n-----END RSA PRIVATE KEY-----")
    result = pack(source, "-PreparedOffline")
    assert result.returncode != 0
    assert "Missing required file" in result.stderr
    target = source / "dist/RealtimeDiffusionArt"
    assert (target / "runtime/python/Lib/module.py").is_file()
    assert (target / "runtime/python/Lib/site-packages/certifi/cacert.pem").is_file()
    assert not (target / "cache/INSTALL_COMPLETE.txt").exists()
    assert {p.relative_to(target / "runtime").as_posix() for p in (target / "runtime").rglob("*") if p.is_file()} == {
        "python/python.exe", "python/Lib/module.py", "python/Lib/site-packages/certifi/cacert.pem"}


def test_prepared_rejects_nonportable_model_path(source: Path) -> None:
    (source / "config.json").write_text(json.dumps({"model_path": "../private-model"}))
    result = pack(source, "-PreparedOffline")
    assert result.returncode != 0
    assert "custom model paths are not portable" in result.stderr
    assert not (source / "dist/RealtimeDiffusionArt/cache/INSTALL_COMPLETE.txt").exists()


@pytest.mark.parametrize("linked_path", ["dist", "assets/outside"])
def test_junction_cannot_copy_or_delete_outside_source(source: Path, tmp_path: Path, linked_path: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")
    link = source / linked_path
    escaped_link = str(link).replace("'", "''")
    escaped_outside = str(outside).replace("'", "''")
    result = subprocess.run([str(POWERSHELL), "-NoProfile", "-Command",
                             f"New-Item -ItemType Junction -Path '{escaped_link}' -Target '{escaped_outside}' | Out-Null"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    try:
        result = pack(source, "-SkipVerify")
        assert result.returncode != 0
        assert "Refusing reparse point" in result.stderr
        assert sentinel.read_text() == "keep"
    finally:
        # rmdir unlinks the junction itself; no recursive traversal.
        link.rmdir()


def test_marker_follows_copied_runtime_and_generation_verification() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    runtime_check = text.index("& $packedPython -I -c")
    generation = text.index("& $packedPython -I (Join-Path $Target 'tools\\verify_offline.py') --root $Target")
    failed = text.index("if ($LASTEXITCODE -ne 0)", generation)
    marker = text.index("Set-Content -LiteralPath (Join-Path $Target 'cache\\INSTALL_COMPLETE.txt')")
    assert runtime_check < generation < failed < marker
    assert "Copy-PackageTree 'runtime\\python'" in text
    assert "Copy-PackageTree 'models'" not in text
