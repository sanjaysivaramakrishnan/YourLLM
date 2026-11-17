"""Unit tests verifying Dockerfile, entrypoint script, and .dockerignore configuration."""

from __future__ import annotations

from pathlib import Path


def test_dockerfile_configuration() -> None:
    """Verify Dockerfile meets all M4 specification and architecture requirements."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must be present in repository root"

    content = dockerfile_path.read_text(encoding="utf-8")

    # 1. Base image pinned by tag and official sha256 digest (DECISIONS.md §4)
    expected_image = (
        "FROM vllm/vllm-openai:v0.7.3@sha256:4f4037303e8c7b69439db1077bb849a0823517c0f785b894dc8e96d58ef3a0c2"
    )
    assert expected_image in content

    # 2. Process supervisor: tini must be installed and used in entrypoint
    assert "tini" in content
    assert 'ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]' in content

    # 3. Architecture Invariant 1: Isolated gateway venv at /opt/gateway
    assert "/opt/gateway" in content
    assert "python3 -m venv /opt/gateway" in content

    # 4. Storage volume and port exposure
    assert "VOLUME /data" in content
    assert "EXPOSE 8000" in content

    # 5. Healthcheck uses liveness probe only
    assert "HEALTHCHECK" in content
    assert "http://127.0.0.1:8000/health/live" in content


def test_docker_entrypoint_script() -> None:
    """Verify docker-entrypoint.sh creates persistent directories and forwards signals."""
    entrypoint_path = Path("docker-entrypoint.sh")
    assert entrypoint_path.exists(), "docker-entrypoint.sh must be present in repository root"

    content = entrypoint_path.read_text(encoding="utf-8")

    # Must be bash with strict error handling
    assert content.startswith("#!/bin/bash")
    assert "set -e" in content

    # Must create database and models subdirectories
    assert "database" in content
    assert "models/huggingface" in content

    # Must use exec to preserve signal forwarding to tini/gateway
    assert 'exec "$@"' in content


def test_dockerignore_exclusions() -> None:
    """Verify .dockerignore excludes tests, caches, virtualenvs, and databases."""
    dockerignore_path = Path(".dockerignore")
    assert dockerignore_path.exists(), ".dockerignore must be present in repository root"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = {line.strip() for line in content.splitlines() if line.strip()}

    # Crucial exclusions
    assert ".venv/" in lines or ".venv" in lines
    assert "tests/" in lines or "tests" in lines
    assert "__pycache__/" in lines or "__pycache__" in lines
    assert "*.db" in lines

