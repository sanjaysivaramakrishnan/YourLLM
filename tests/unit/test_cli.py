"""Tests for yourllm-cli command line interface using Typer CliRunner."""

import re
from pathlib import Path

from typer.testing import CliRunner

from yourllm.cli.main import app

runner = CliRunner()


def test_cli_key_create_and_list(tmp_path: Path) -> None:
    """Verify key creation displays plaintext once, and list never displays plaintext or hashes."""
    db_file = tmp_path / "cli_test.db"

    # 1. Create Key
    create_res = runner.invoke(
        app,
        ["key", "create", "--name", "production-service", "--db", str(db_file)],
    )
    assert create_res.exit_code == 0
    assert "API Key Created Successfully" in create_res.stdout
    assert "production-service" in create_res.stdout

    # Extract the generated key from create output
    match = re.search(r"(yllm_[a-zA-Z0-9_-]+)", create_res.stdout)
    assert match is not None
    plaintext_key = match.group(1)
    key_id = plaintext_key.split("_")[1]

    # 2. List Keys
    list_res = runner.invoke(app, ["key", "list", "--db", str(db_file)])
    assert list_res.exit_code == 0
    assert key_id in list_res.stdout
    assert "production-service" in list_res.stdout
    assert "active" in list_res.stdout

    # Critical requirement: Plaintext key and hashes must NEVER appear in list output
    assert plaintext_key not in list_res.stdout
    assert "sha256" not in list_res.stdout


def test_cli_key_rotate_and_revoke(tmp_path: Path) -> None:
    """Verify key rotation displays new plaintext and revocation deactivates key."""
    db_file = tmp_path / "cli_rotate.db"

    # 1. Create Key
    res_create = runner.invoke(
        app,
        ["key", "create", "--name", "agent-key", "--db", str(db_file)],
    )
    assert res_create.exit_code == 0
    match = re.search(r"(yllm_[a-zA-Z0-9_-]+)", res_create.stdout)
    assert match is not None
    orig_key = match.group(1)
    orig_id = orig_key.split("_")[1]

    # 2. Rotate Key
    res_rotate = runner.invoke(
        app,
        ["key", "rotate", orig_id, "--db", str(db_file)],
    )
    assert res_rotate.exit_code == 0
    assert "successfully rotated" in res_rotate.stdout
    assert orig_id in res_rotate.stdout
    assert "revoked" in res_rotate.stdout

    match_new = re.search(r"(yllm_[a-zA-Z0-9_-]+)", res_rotate.stdout)
    assert match_new is not None
    new_key = match_new.group(1)
    new_id = new_key.split("_")[1]
    assert new_key != orig_key

    # 3. Revoke New Key
    res_revoke = runner.invoke(
        app,
        ["key", "revoke", new_id, "--db", str(db_file)],
    )
    assert res_revoke.exit_code == 0
    assert f"Key '{new_id}' has been revoked" in res_revoke.stdout

    # 4. List Keys to verify both are revoked
    res_list = runner.invoke(app, ["key", "list", "--db", str(db_file)])
    assert res_list.exit_code == 0
    assert "revoked" in res_list.stdout
