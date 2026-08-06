"""Static contract tests for the Windows setup scripts (CLAUDE.md T-0.5).

The build host is Linux, so ``scripts/setup_env.ps1`` and
``scripts/pull_models.ps1`` cannot be executed here and PowerShell is not
installed. Their correctness is pinned instead by asserting the properties that
make them safe, portable, and idempotent on the Windows target:

* no download runs without an existence check in front of it, which is what
  makes a second run a no-op,
* no absolute user path is baked in (CLAUDE.md section 0.6),
* nothing is installed system wide without the user typing the command,
* and, most importantly, the tier to model mapping in ``pull_models.ps1`` is
  parsed out of the script text and compared against ``TIER_PROFILES``, the
  single source of truth in ``jarvis.config``. A drifted mapping would pull the
  wrong model for the machine, and nothing else in the suite would notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jarvis.config import TIER_PROFILES, Tier, ToolsConfig, TtsConfig, WakeConfig

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_env.ps1"
PULL = ROOT / "scripts" / "pull_models.ps1"
SCRIPTS = (SETUP, PULL)

# Written as escapes so this checker does not trip over its own source.
EM_DASH = "\u2014"
EN_DASH = "\u2013"

#: Statements that pull bytes off the network.
DOWNLOAD_MARKERS = (
    "Invoke-WebRequest",
    "Invoke-RestMethod -OutFile",
    "Start-BitsTransfer",
    "-OutFile",
    "curl.exe",
    "ollama pull",
    "hf_hub_download",
    "snapshot_download",
)

#: Checks that make a download conditional, and therefore idempotent.
GUARD_MARKERS = ("Test-Path", "Test-OllamaModelPresent", "ollama list")

#: How many lines above a download a guard may sit and still count.
GUARD_WINDOW = 12

MAX_LINE_LENGTH = 100


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments(text: str) -> list[str]:
    """Blank out comment lines, keeping line numbering intact.

    Handles ``#`` line comments and ``<# ... #>`` block comments, which is all
    the two scripts use. Trailing comments on a code line are left alone; no
    marker string appears in one.
    """
    out: list[str] = []
    in_block = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if in_block:
            if "#>" in stripped:
                in_block = False
            out.append("")
            continue
        if stripped.startswith("<#"):
            if "#>" not in stripped:
                in_block = True
            out.append("")
            continue
        if stripped.startswith("#"):
            out.append("")
            continue
        out.append(raw)
    return out


def _parse_ps_hashtable(text: str, name: str) -> list[tuple[str, str]]:
    """Pull a ``$Name = @{ 'k' = 'v' ... }`` literal out of PowerShell source."""
    pattern = r"\$" + name + r"\s*=\s*@\{(.*?)^\}"
    block = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if block is None:
        pytest.fail(f"${name} hashtable literal not found")
    return re.findall(r"'([^']+)'\s*=\s*'([^']+)'", block.group(1))


def _parse_ps_array(text: str, name: str) -> list[str]:
    """Pull a ``$Name = @('a', 'b')`` literal out of PowerShell source."""
    block = re.search(r"\$" + name + r"\s*=\s*@\((.*?)\)", text, re.DOTALL)
    if block is None:
        pytest.fail(f"${name} array literal not found")
    return re.findall(r"'([^']+)'", block.group(1))


# ---------------------------------------------------------------------------
# Existence and shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_exists_and_is_not_a_stub(script: Path) -> None:
    assert script.is_file(), f"{script} is missing"
    text = _read(script)
    assert len(text.strip()) > 1000, f"{script.name} looks like a placeholder"
    code = [line for line in _strip_comments(text) if line.strip()]
    assert len(code) > 60, f"{script.name} has only {len(code)} lines of real code"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_stops_on_the_first_error(script: Path) -> None:
    text = _read(script)
    assert re.search(
        r"\$ErrorActionPreference\s*=\s*['\"]Stop['\"]", text
    ), f"{script.name} must set $ErrorActionPreference = 'Stop'"
    assert "Set-StrictMode" in text, f"{script.name} should run under Set-StrictMode"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_declares_a_param_block(script: Path) -> None:
    text = _read(script)
    assert re.search(r"^param\s*\(", text, re.MULTILINE | re.IGNORECASE), (
        f"{script.name} must declare a param() block at the top level"
    )
    assert "[CmdletBinding()]" in text, f"{script.name} should be an advanced script"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_has_comment_based_help(script: Path) -> None:
    text = _read(script)
    assert text.lstrip().startswith("<#"), f"{script.name} must open with a help block"
    for section in (".SYNOPSIS", ".DESCRIPTION", ".PARAMETER", ".EXAMPLE"):
        assert section in text, f"{script.name} help is missing {section}"


def test_setup_declares_the_documented_switches() -> None:
    text = _read(SETUP)
    block = re.search(r"^param\s*\((.*?)^\)", text, re.DOTALL | re.MULTILINE)
    assert block is not None
    for switch in ("SkipModels", "SkipHud", "Force"):
        assert re.search(rf"\[switch\]\s*\${switch}\b", block.group(1)), (
            f"setup_env.ps1 must expose a -{switch} switch"
        )


def test_pull_models_declares_the_vision_switch() -> None:
    text = _read(PULL)
    block = re.search(r"^param\s*\((.*?)^\)", text, re.DOTALL | re.MULTILINE)
    assert block is not None
    assert re.search(r"\[switch\]\s*\$WithVision\b", block.group(1)), (
        "pull_models.ps1 must expose a -WithVision switch"
    )


# ---------------------------------------------------------------------------
# Style and portability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_em_or_en_dashes(script: Path) -> None:
    """CLAUDE.md section 5. Also enforced repo wide by tests/test_style.py."""
    offenders = [
        f"{script.name}:{number}: {line.strip()[:70]}"
        for number, line in enumerate(_read(script).splitlines(), start=1)
        if EM_DASH in line or EN_DASH in line
    ]
    assert not offenders, "dashes found:\n" + "\n".join(offenders)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_pure_ascii(script: Path) -> None:
    """Windows PowerShell 5.1 reads .ps1 files as ANSI unless there is a BOM."""
    text = _read(script)
    assert text.isascii(), f"{script.name} contains non-ASCII characters"
    assert not text.startswith("\ufeff"), f"{script.name} must not carry a BOM"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_hardcoded_absolute_user_paths(script: Path) -> None:
    """CLAUDE.md section 0.6: nothing machine specific in the committed tree."""
    # A drive-rooted path such as C:\Users\... or D:/models. The lookbehind keeps
    # URL schemes ("https://") from matching.
    drive_rooted = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
    offenders: list[str] = []
    for number, line in enumerate(_read(script).splitlines(), start=1):
        lowered = line.lower()
        if (
            drive_rooted.search(line)
            or "c:\\users" in lowered
            or "/home/" in lowered
            or "/users/" in lowered
        ):
            offenders.append(f"{script.name}:{number}: {line.strip()[:70]}")
    assert not offenders, "absolute user paths found:\n" + "\n".join(offenders)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_paths_are_derived_from_the_script_location(script: Path) -> None:
    text = _read(script)
    assert "$PSScriptRoot" in text, f"{script.name} must locate the repo from itself"
    assert "Join-Path" in text, f"{script.name} should compose paths with Join-Path"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_lines_stay_within_the_project_limit(script: Path) -> None:
    offenders = [
        f"{script.name}:{number}: {len(line)} chars"
        for number, line in enumerate(_read(script).splitlines(), start=1)
        if len(line) > MAX_LINE_LENGTH
    ]
    assert not offenders, "lines over the limit:\n" + "\n".join(offenders)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_brackets_are_balanced(script: Path) -> None:
    """Cheap guard against a truncated or half-edited script.

    Every brace, parenthesis, and bracket in both files is balanced inside its
    own string literal too, so a plain count is meaningful here.
    """
    text = _read(script)
    for opener, closer in ("{}", "()", "[]"):
        assert text.count(opener) == text.count(closer), (
            f"{script.name} has unbalanced {opener}{closer}: "
            f"{text.count(opener)} open, {text.count(closer)} close"
        )


# ---------------------------------------------------------------------------
# Safety: no elevation, no silent system-wide installs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_never_requests_elevation(script: Path) -> None:
    """CLAUDE.md section 6: only jarvis-helper.exe is ever elevated."""
    lowered = _read(script).lower()
    assert "#requires -runasadministrator" not in lowered
    assert "-verb runas" not in lowered
    assert "runasadministrator" not in lowered


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_never_invokes_a_package_manager(script: Path) -> None:
    """Install commands are printed for the user, never executed for them."""
    offenders: list[str] = []
    invocation = re.compile(
        r"^\s*(&\s*)?(winget|choco|scoop|pip|pip3)\s+install\b", re.IGNORECASE
    )
    for number, line in enumerate(_strip_comments(_read(script)), start=1):
        if invocation.match(line) or "Start-Process winget" in line:
            offenders.append(f"{script.name}:{number}: {line.strip()[:70]}")
    assert not offenders, "package manager invoked directly:\n" + "\n".join(offenders)


def test_setup_reports_install_commands_for_every_prerequisite() -> None:
    """Each checked tool comes with the exact command that installs it."""
    text = _read(SETUP)
    for tool, command in (
        ("uv", "astral-sh.uv"),
        ("ollama", "Ollama.Ollama"),
        ("rust", "Rustlang.Rustup"),
        ("node", "OpenJS.NodeJS"),
    ):
        assert f"-Name '{tool}'" in text or f"Test-CommandExists -Name '{tool}'" in text, (
            f"setup_env.ps1 does not check for {tool}"
        )
        assert command in text, f"setup_env.ps1 gives no install command for {tool}"
    assert "winget install" in text


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_deletions_are_never_recursive(script: Path) -> None:
    """A setup script must never be able to wipe a tree."""
    for number, line in enumerate(_strip_comments(_read(script)), start=1):
        if "Remove-Item" in line:
            assert "-Recurse" not in line, (
                f"{script.name}:{number} uses a recursive delete: {line.strip()[:70]}"
            )


# ---------------------------------------------------------------------------
# Idempotency: every download is guarded
# ---------------------------------------------------------------------------


def _download_sites(script: Path) -> list[tuple[int, str]]:
    lines = _strip_comments(_read(script))
    return [
        (number, line)
        for number, line in enumerate(lines, start=1)
        if any(marker in line for marker in DOWNLOAD_MARKERS)
    ]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_download_is_guarded_by_an_existence_check(script: Path) -> None:
    """The guard in front of each fetch is what makes a second run a no-op."""
    lines = _strip_comments(_read(script))
    offenders: list[str] = []
    for number, line in _download_sites(script):
        start = max(0, number - 1 - GUARD_WINDOW)
        window = lines[start:number]
        if not any(guard in text for text in window for guard in GUARD_MARKERS):
            offenders.append(f"{script.name}:{number}: {line.strip()[:70]}")
    assert not offenders, (
        "unguarded downloads, add a Test-Path or an 'ollama list' check "
        f"within {GUARD_WINDOW} lines:\n" + "\n".join(offenders)
    )


def test_the_download_guard_check_is_not_vacuous() -> None:
    """Fail loudly if the download detection stops finding anything."""
    sites = _download_sites(PULL)
    assert len(sites) >= 3, (
        "expected pull_models.ps1 to contain download statements, "
        f"found {len(sites)}. The marker list is probably stale."
    )
    text = _read(PULL)
    assert "Test-Path" in text
    assert "Test-OllamaModelPresent" in text
    # Every artifact flows through the one guarded helper, so the declared
    # download table is the real measure of coverage.
    artifacts = re.findall(r"Relative\s*=\s*'([^']+)'", text)
    assert len(artifacts) >= 6, f"only {len(artifacts)} artifacts declared: {artifacts}"
    assert len(artifacts) == len(set(artifacts)), "duplicate download destinations"
    assert "foreach ($item in $FileDownloads)" in text
    urls = re.findall(r"Urls\s*=\s*@\(", text)
    assert len(urls) == len(artifacts), "every artifact needs at least one source URL"


def test_downloads_land_in_the_models_directory() -> None:
    """Model artifacts are gitignored under models/, never loose in the tree."""
    text = _read(PULL)
    assert re.search(r"\$ModelsDir\s*=\s*Join-Path \$RepoRoot 'models'", text)
    assert "Join-Path $ModelsDir" in text


def test_partial_downloads_do_not_masquerade_as_finished_ones() -> None:
    """Bytes land in a .partial file and a short file is refetched."""
    text = _read(PULL)
    assert ".partial" in text, "download to a temp name, then move into place"
    assert "MinimumBytes" in text, "a truncated or error-page file must be rejected"
    assert "Move-Item" in text


def test_setup_seeds_config_yaml_only_when_absent() -> None:
    """config.yaml holds the user's machine specifics. Never overwrite it."""
    lines = _strip_comments(_read(SETUP))
    copies = [(n, line) for n, line in enumerate(lines, start=1) if "Copy-Item" in line]
    assert len(copies) == 1, f"expected exactly one Copy-Item, found {len(copies)}"
    number, line = copies[0]
    assert "config.yaml" not in line or "$UserConfig" in line
    assert "-Force" not in line, "copying with -Force would clobber the user's config"
    window = lines[max(0, number - 1 - 8) : number]
    assert any("Test-Path" in text for text in window), (
        "the config copy must be guarded by a Test-Path"
    )
    assert "config.example.yaml" in _read(SETUP)


def test_setup_creates_the_runtime_directories() -> None:
    text = _read(SETUP)
    block = re.search(r"\$RuntimeDirs\s*=\s*@\((.*?)\n\)", text, re.DOTALL)
    assert block is not None, "$RuntimeDirs not found"
    names = set(re.findall(r"Join-Path \$RepoRoot '([^']+)'", block.group(1)))
    assert {"data", "logs", "models"} <= names, f"missing runtime dirs, got {names}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_directory_creation_is_idempotent(script: Path) -> None:
    """New-Item -Force succeeds when the directory already exists."""
    for number, line in enumerate(_strip_comments(_read(script)), start=1):
        if "New-Item" in line and "Directory" in line:
            assert "-Force" in line, (
                f"{script.name}:{number} would fail on a second run: {line.strip()[:70]}"
            )


def test_setup_resolves_the_hardware_tier_and_can_skip_the_work() -> None:
    text = _read(SETUP)
    assert "verify_cuda.py" in text, "setup must resolve and persist the hardware tier"
    assert "uv run python" in text
    assert "Get-ConfiguredTier" in text, "an already resolved tier should be reused"
    assert "$Force" in text


def test_setup_hands_off_to_pull_models_unless_skipped() -> None:
    text = _read(SETUP)
    assert "pull_models.ps1" in text
    assert "$SkipModels" in text
    assert "& $PullModels" in text


# ---------------------------------------------------------------------------
# Dependency install surface
# ---------------------------------------------------------------------------


def test_setup_syncs_every_extra_and_the_dev_group() -> None:
    """CLAUDE.md T-0.5 names the exact extras a full Windows install needs."""
    text = _read(SETUP)
    block = re.search(r"\$UvSyncArgs\s*=\s*@\((.*?)\n\)", text, re.DOTALL)
    assert block is not None, "$UvSyncArgs not found"
    body = block.group(1)
    assert re.search(r"'sync'", body)
    for extra in ("audio", "stt", "tts", "gpu", "windows", "vision"):
        assert re.search(rf"'--extra',\s*'{extra}'", body), f"--extra {extra} missing"
    assert re.search(r"'--group',\s*'dev'", body), "--group dev missing"


def test_setup_extras_exist_in_pyproject() -> None:
    """A typo in an extra name would fail only on the Windows host."""
    text = _read(SETUP)
    block = re.search(r"\$UvSyncArgs\s*=\s*@\((.*?)\n\)", text, re.DOTALL)
    assert block is not None
    requested = re.findall(r"'--extra',\s*'([^']+)'", block.group(1))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(
        r"\[project\.optional-dependencies\](.*?)^\[", pyproject, re.DOTALL | re.MULTILINE
    )
    assert declared is not None, "no optional-dependencies table in pyproject.toml"
    names = set(re.findall(r"^(\w[\w-]*)\s*=\s*\[", declared.group(1), re.MULTILINE))
    assert set(requested) <= names, f"unknown extras {set(requested) - names}"


# ---------------------------------------------------------------------------
# The mapping that actually matters
# ---------------------------------------------------------------------------


def test_tier_to_model_mapping_matches_tier_profiles_exactly() -> None:
    """Parse $TierModels out of pull_models.ps1 and diff it against Python.

    This is the test that earns its keep. The script cannot import
    ``jarvis.config``, so the mapping is duplicated in PowerShell; if the two
    ever drift, a machine gets a model that does not fit its VRAM.
    """
    pairs = _parse_ps_hashtable(_read(PULL), "TierModels")
    assert pairs, "$TierModels parsed as empty, the literal shape must have changed"

    keys = [key for key, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate tier keys in the script: {keys}"

    parsed = dict(pairs)
    expected = {tier.value: profile.llm_model for tier, profile in TIER_PROFILES.items()}
    assert parsed == expected, (
        "pull_models.ps1 disagrees with jarvis.config.TIER_PROFILES.\n"
        f"  script: {parsed}\n"
        f"  python: {expected}"
    )


def test_every_tier_except_auto_is_covered() -> None:
    parsed = dict(_parse_ps_hashtable(_read(PULL), "TierModels"))
    assert set(parsed) == {tier.value for tier in Tier if tier is not Tier.AUTO}
    assert Tier.AUTO.value not in parsed, "'auto' is not a downloadable tier"


def test_tier_validate_set_matches_the_mapping() -> None:
    """-Tier must accept exactly the tiers the script knows how to serve."""
    text = _read(PULL)
    validate = re.search(r"\[ValidateSet\((.*?)\)\]", text, re.DOTALL)
    assert validate is not None, "the -Tier parameter needs a ValidateSet"
    allowed = set(re.findall(r"'([^']+)'", validate.group(1)))
    assert allowed == set(dict(_parse_ps_hashtable(text, "TierModels")))


def test_vision_tiers_match_tier_profile_support() -> None:
    """The VLM needs about 6 GB spare, so cpu and gpu-6 must be excluded."""
    parsed = set(_parse_ps_array(_read(PULL), "VisionTiers"))
    expected = {
        tier.value for tier, profile in TIER_PROFILES.items() if profile.supports_vision
    }
    assert parsed == expected, (
        f"vision tiers drifted.\n  script: {sorted(parsed)}\n  python: {sorted(expected)}"
    )
    assert Tier.CPU.value not in parsed
    assert Tier.GPU_6.value not in parsed


def test_vision_model_matches_the_configured_default() -> None:
    match = re.search(r"\$VisionModel\s*=\s*'([^']+)'", _read(PULL))
    assert match is not None, "$VisionModel not found"
    assert match.group(1) == ToolsConfig().vision_model


def test_vision_pull_is_opt_in_and_tier_gated() -> None:
    text = _read(PULL)
    assert "$WithVision" in text
    assert "$VisionTiers -contains" in text, "the vision pull must be tier gated"


def test_downloaded_artifacts_match_the_configured_wake_word_and_voice() -> None:
    """The files fetched are the ones the runtime config actually asks for."""
    text = _read(PULL)
    assert f"{WakeConfig().model}_v0.1.onnx" in text, "wake word model not downloaded"
    assert f"{TtsConfig().voice}.pt" in text, "configured Kokoro voice not downloaded"


def test_the_phonemiser_model_is_prefetched() -> None:
    """Kokoro's G2P stage downloads a spaCy model the first time it runs.

    misaki.en.G2P calls spacy.cli.download when en_core_web_sm is absent. Left
    to happen on its own that lands on the first sentence JARVIS ever speaks:
    roughly twenty seconds of silence, and an outright failure if the machine
    is offline by then. Setup has to pay that cost instead.
    """
    text = _read(PULL)
    assert "en_core_web_sm" in text, "the spaCy model is never pre-fetched"
    assert "spacy download" in text, "no download command for it"


def test_the_phonemiser_prefetch_is_idempotent() -> None:
    """A second run must not reinstall it. The script promises idempotency."""
    text = _read(PULL)
    assert "find_spec('en_core_web_sm')" in text, "no presence check before installing"


def test_wake_word_feature_models_and_vad_are_downloaded() -> None:
    """openWakeWord is useless without its two shared feature extractors."""
    text = _read(PULL)
    for artifact in ("melspectrogram.onnx", "embedding_model.onnx", "silero_vad.onnx"):
        assert artifact in text, f"{artifact} is never downloaded"
    assert "kokoro" in text.lower(), "Kokoro weights are never downloaded"


def test_download_sources_are_permissively_licensed_projects() -> None:
    """CLAUDE.md sections 1 and 3: only the locked stack, only local hosts."""
    text = _read(PULL)
    assert "dscripka/openWakeWord" in text
    assert "snakers4/silero-vad" in text
    assert "hexgrad/Kokoro-82M" in text
    for rejected in ("coqui", "xtts", "f5-tts", "fish-speech", "picovoice", "porcupine"):
        assert rejected not in text.lower(), f"rejected component referenced: {rejected}"


def test_pull_models_reports_what_it_skipped_versus_downloaded() -> None:
    text = _read(PULL)
    for bucket in ("$script:Downloaded", "$script:Skipped", "$script:Failed"):
        assert bucket in text, f"{bucket} tally missing from the summary"
    assert "Already present:" in text
    assert "exit 1" in text, "a failed download must be reported through the exit code"


class TestTheDownloaderAndTheLoadersAgree:
    """Every directory the script writes into must be one the code looks in.

    Two real first-run failures came from exactly this mismatch and nothing
    caught either. pull_models.ps1 put the Silero checkpoint in models\\silero
    while vad.py searched models\\silero_vad and models\\vad, so a correctly
    downloaded model reported itself missing and told the reader to run the
    script that had already downloaded it. openWakeWord's feature extractors
    had the same shape of problem.
    """

    @staticmethod
    def _download_dirs() -> set[str]:
        """First path segment of every Relative= entry in the script."""
        text = _read(PULL)
        found = set()
        for match in re.finditer(r"Relative\s*=\s*'([^']+)'", text):
            parts = match.group(1).replace("/", "\\").split("\\")
            if len(parts) > 1:
                found.add(parts[0])
        return found

    def test_the_script_declares_the_directories_we_expect(self) -> None:
        """Guards the parser itself, so a rename cannot make this class vacuous."""
        assert self._download_dirs() >= {"openwakeword", "silero", "kokoro"}

    def test_every_silero_download_directory_is_searched_by_the_vad(
        self, tmp_path: Path
    ) -> None:
        from jarvis.audio.vad import SileroVad
        from jarvis.config import load_config

        config = load_config(tmp_path / "absent.yaml", paths={"models_dir": str(tmp_path)})
        detector = SileroVad(config)

        for directory in self._download_dirs():
            if "silero" not in directory:
                continue
            target = tmp_path / directory / "silero_vad.onnx"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"stub")
            found = detector.find_model_file()
            target.unlink()
            assert found is not None, (
                f"pull_models.ps1 downloads into models\\{directory}, "
                "which SileroVad.find_model_file never looks in"
            )

    def test_every_wake_download_directory_is_searched_by_the_detector(
        self, tmp_path: Path
    ) -> None:
        from jarvis.audio.wake import WakeWordDetector
        from jarvis.config import load_config

        config = load_config(tmp_path / "absent.yaml", paths={"models_dir": str(tmp_path)})
        detector = WakeWordDetector(config)

        for directory in self._download_dirs():
            if "wake" not in directory:
                continue
            root = tmp_path / directory
            root.mkdir(parents=True, exist_ok=True)
            (root / "hey_jarvis_v0.1.onnx").write_bytes(b"stub")
            (root / "melspectrogram.onnx").write_bytes(b"stub")
            (root / "embedding_model.onnx").write_bytes(b"stub")

            assert detector._find_model_file() is not None, (
                f"the wake model is downloaded into models\\{directory}, which is not searched"
            )
            features = detector._find_feature_models()
            assert len(features) == 2, (
                f"the feature extractors are downloaded into models\\{directory}, "
                f"but only {sorted(features)} were found"
            )
