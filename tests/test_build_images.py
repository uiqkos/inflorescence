"""`inflorescence build-images`: the packaged Dockerfiles and the local-build path.

Everything here runs without Docker — subprocess is always stubbed. What is being pinned
down is the contract: the images are built from Dockerfiles that ship inside the package
(so a wheel install can build them), and they are tagged with exactly the references the
docker rung looks for — a tag under any other name would build an image nothing ever runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def _ns(**overrides) -> argparse.Namespace:
    defaults = {"language": "all", "registry": "ghcr.io/uiqkos", "tag": "v1"}
    return argparse.Namespace(**{**defaults, **overrides})


def test_parser_defaults_match_the_expected_image_names() -> None:
    from inflorescence.__main__ import build_parser

    args = build_parser().parse_args(["build-images"])
    assert args.tag == "v1"
    assert args.registry == "ghcr.io/uiqkos"
    assert args.language == "all"


def test_dockerfiles_ship_inside_the_package() -> None:
    """From a wheel there is no repository checkout to fall back to — the files must be
    importable package data, not paths relative to a git root."""
    from inflorescence.__main__ import packaged_dockerfile

    for name in ("scip-go", "scip-node"):
        dockerfile = packaged_dockerfile(name)
        assert dockerfile.is_file(), f"{dockerfile} missing from the package"
        assert "src/inflorescence/docker" in str(dockerfile) or "inflorescence/docker" in str(dockerfile)


def test_packaged_dockerfiles_pin_indexer_versions() -> None:
    """@latest would make "build it yourself and compare" meaningless. CI runs this from
    the committed tree, so a pin that exists only in a working copy fails here."""
    from inflorescence.__main__ import packaged_dockerfile

    go = packaged_dockerfile("scip-go").read_text()
    assert "/scip-go@v" in go, "scip-go install is not version-pinned"
    node = packaged_dockerfile("scip-node").read_text()
    assert "@sourcegraph/scip-typescript@" in node, "scip-typescript is not version-pinned"
    assert "@sourcegraph/scip-python@" in node, "scip-python is not version-pinned"
    for text, name in ((go, "scip-go"), (node, "scip-node")):
        # comments may discuss @latest; the instructions themselves must not use it
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        assert "@latest" not in code, f"{name}.Dockerfile still installs @latest"


def test_build_argv_tags_exactly_what_the_docker_rung_expects() -> None:
    from inflorescence.__main__ import build_image_argv
    from inflorescence.code_indexer import scip_semantic as ss

    go_argv = build_image_argv("scip-go", registry="ghcr.io/uiqkos", tag="v1")
    node_argv = build_image_argv("scip-node", registry="ghcr.io/uiqkos", tag="v1")

    assert go_argv[go_argv.index("-t") + 1] == ss._DEFAULT_IMAGE_TAGS["go"]
    node_ref = node_argv[node_argv.index("-t") + 1]
    assert node_ref == ss._DEFAULT_IMAGE_TAGS["python"] == ss._DEFAULT_IMAGE_TAGS["typescript"]


def test_cmd_build_images_builds_both_images_by_default(monkeypatch) -> None:
    from inflorescence import __main__ as m

    calls: list[list[str]] = []
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        m.subprocess, "run", lambda argv, **k: (calls.append(argv), SimpleNamespace(returncode=0))[1]
    )

    assert m._cmd_build_images(_ns()) == 0
    tags = [c[c.index("-t") + 1] for c in calls]
    assert tags == [
        "ghcr.io/uiqkos/inflorescence-scip-go:v1",
        "ghcr.io/uiqkos/inflorescence-scip-node:v1",
    ]
    for c in calls:
        assert c[:2] == ["docker", "build"]
        assert Path(c[c.index("-f") + 1]).is_file()


def test_cmd_build_images_language_filter_builds_one_image(monkeypatch) -> None:
    from inflorescence import __main__ as m

    calls: list[list[str]] = []
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        m.subprocess, "run", lambda argv, **k: (calls.append(argv), SimpleNamespace(returncode=0))[1]
    )

    assert m._cmd_build_images(_ns(language="go")) == 0
    assert len(calls) == 1
    assert "ghcr.io/uiqkos/inflorescence-scip-go:v1" in calls[0]


def test_cmd_build_images_honors_registry_and_tag_overrides(monkeypatch) -> None:
    from inflorescence import __main__ as m

    calls: list[list[str]] = []
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        m.subprocess, "run", lambda argv, **k: (calls.append(argv), SimpleNamespace(returncode=0))[1]
    )

    assert m._cmd_build_images(_ns(language="node", registry="example.com/me", tag="dev")) == 0
    assert "example.com/me/inflorescence-scip-node:dev" in calls[0]


def test_cmd_build_images_fails_cleanly_without_docker(monkeypatch, capsys) -> None:
    from inflorescence import __main__ as m

    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    ran: list[list[str]] = []
    monkeypatch.setattr(m.subprocess, "run", lambda argv, **k: ran.append(argv))

    assert m._cmd_build_images(_ns()) == 1
    assert ran == [], "no docker on PATH must mean no subprocess at all"
    assert "docker not found" in capsys.readouterr().err


def test_cmd_build_images_stops_on_first_failed_build(monkeypatch) -> None:
    from inflorescence import __main__ as m

    calls: list[list[str]] = []
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        m.subprocess, "run", lambda argv, **k: (calls.append(argv), SimpleNamespace(returncode=3))[1]
    )

    assert m._cmd_build_images(_ns()) == 3
    assert len(calls) == 1, "a failed build must not be followed by the next one"


def test_locally_built_tag_wins_over_a_digest_pin(monkeypatch, tmp_path: Path) -> None:
    """The point of `build-images`: once the tag exists locally, the docker rung must use
    it — a digest reference always names the registry's bytes and would trigger a pull."""
    from inflorescence.code_indexer import scip_semantic as ss
    from inflorescence.code_indexer.models import IndexerConfig

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    digest = "ghcr.io/uiqkos/inflorescence-scip-go@sha256:" + "0" * 64
    monkeypatch.setattr(ss, "_DEFAULT_IMAGES", {**ss._DEFAULT_IMAGES, "go": digest})
    monkeypatch.setattr(ss, "_local_image_present", lambda image: True)

    invocation = ss.resolve_scip_invocation("go", tmp_path, tmp_path, IndexerConfig())
    assert invocation is not None
    assert ss._DEFAULT_IMAGE_TAGS["go"] in invocation.argv
    assert digest not in invocation.argv


def test_digest_pin_is_used_when_no_local_tag_exists(monkeypatch, tmp_path: Path) -> None:
    from inflorescence.code_indexer import scip_semantic as ss
    from inflorescence.code_indexer.models import IndexerConfig

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    digest = "ghcr.io/uiqkos/inflorescence-scip-go@sha256:" + "0" * 64
    monkeypatch.setattr(ss, "_DEFAULT_IMAGES", {**ss._DEFAULT_IMAGES, "go": digest})
    monkeypatch.setattr(ss, "_local_image_present", lambda image: False)

    invocation = ss.resolve_scip_invocation("go", tmp_path, tmp_path, IndexerConfig())
    assert invocation is not None
    assert digest in invocation.argv


def test_default_pins_are_digests_and_tags_are_tags() -> None:
    """The pull path must be immutable (a registry tag can be re-pointed, a digest cannot);
    the human path must stay a tag (it is what `build-images` creates and docs show)."""
    from inflorescence.code_indexer import scip_semantic as ss

    for language, pin in ss._DEFAULT_IMAGES.items():
        assert "@sha256:" in pin, f"{language} pin is not a digest reference: {pin}"
    for language, tag in ss._DEFAULT_IMAGE_TAGS.items():
        assert "@" not in tag and tag.endswith(":v1"), f"{language} tag is not a plain tag: {tag}"
        # digest and tag must name the same repository, or the local-tag preference
        # would silently switch to a different image
        assert ss._DEFAULT_IMAGES[language].split("@")[0] == tag.split(":")[0]


def test_image_availability_dedups_the_shared_node_image(monkeypatch) -> None:
    """python and typescript run the same image; `doctor` must show one row for it, and
    the go image separately, each with its locality."""
    from inflorescence.code_indexer import scip_semantic as ss
    from inflorescence.code_indexer.models import IndexerConfig

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    monkeypatch.setattr(ss, "_local_image_present", lambda image: True)

    report = {s.image: s for s in ss.image_availability(IndexerConfig())}
    assert len(report) == 2
    node = report[ss._DEFAULT_IMAGE_TAGS["python"]]
    assert sorted(node.languages) == ["python", "typescript"]
    assert node.local is True
    assert report[ss._DEFAULT_IMAGE_TAGS["go"]].languages == ["go"]


def test_image_availability_is_empty_without_a_docker_rung(monkeypatch) -> None:
    from inflorescence.code_indexer import scip_semantic as ss
    from inflorescence.code_indexer.models import IndexerConfig

    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/local/bin/anything")
    monkeypatch.setattr(ss, "docker_cli_present", lambda: False)

    assert ss.image_availability(IndexerConfig()) == []


def test_scip_images_override_is_never_rewritten_to_a_local_tag(monkeypatch, tmp_path: Path) -> None:
    """`SCIP_IMAGES` is the user's word; the local-tag preference applies only to our own
    default pin."""
    from inflorescence.code_indexer import scip_semantic as ss
    from inflorescence.code_indexer.models import IndexerConfig

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    monkeypatch.setattr(ss, "_local_image_present", lambda image: True)

    config = IndexerConfig(scip_images={"go": "my-registry/scip-go:pinned"})
    invocation = ss.resolve_scip_invocation("go", tmp_path, tmp_path, config)
    assert invocation is not None
    assert "my-registry/scip-go:pinned" in invocation.argv
    assert ss._DEFAULT_IMAGE_TAGS["go"] not in invocation.argv
