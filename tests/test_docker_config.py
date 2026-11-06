"""Tests for the Docker delivery configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_docker_delivery_files() -> None:
    dockerfile = (
        PROJECT_ROOT / "Dockerfile"
    ).read_text(encoding="utf-8")

    dockerignore = (
        PROJECT_ROOT / ".dockerignore"
    ).read_text(encoding="utf-8")

    readme = (
            PROJECT_ROOT / "README.md"
    ).read_text(encoding="utf-8")

    assert (
            "ARG BASE_IMAGE=python:3.12-slim"
            in dockerfile
    )
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "pip install --no-cache-dir ." in dockerfile
    assert "EXPOSE 8765" in dockerfile
    cmd_lines = [
        line
        for line in dockerfile.splitlines()
        if line.startswith("CMD ")
    ]

    assert len(cmd_lines) == 1
    assert "medlit_agent.main:app" in cmd_lines[0]

    assert ".env" in dockerignore
    assert "data/" in dockerignore
    assert "__pycache__/" in dockerignore

    assert "## Run with Docker" in readme
    assert "docker build -t medlit-agent ." in readme
    assert "docker volume create medlit-data" in readme
    assert "--env SILICONFLOW_API_KEY" in readme