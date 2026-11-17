"""YourLLM CLI for local management inside the container."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from yourllm.auth.service import create_key, list_keys, revoke_key, rotate_key
from yourllm.storage.database import Database
from yourllm.storage.migrations import MigrationRunner

app = typer.Typer(
    name="yourllm-cli",
    help="YourLLM management CLI for local administration and key lifecycle.",
    no_args_is_help=True,
)

key_app = typer.Typer(
    name="key",
    help="Manage API keys (create, list, revoke, rotate).",
    no_args_is_help=True,
)
app.add_typer(key_app, name="key")

console = Console()


def get_database(db_path: Path | None = None) -> Database:
    """Resolve database path from argument, environment, or default /data location."""
    if db_path:
        target = db_path
    elif "YLLM_DATABASE_PATH" in os.environ:
        target = Path(os.environ["YLLM_DATABASE_PATH"])
    elif "YLLM_DATA_DIR" in os.environ:
        target = Path(os.environ["YLLM_DATA_DIR"]) / "database" / "yourllm.db"
    else:
        # Default in-container path or local fallback
        default_dir = Path("/data") if Path("/data").exists() else Path("./data")
        target = default_dir / "database" / "yourllm.db"

    db = Database(target)
    # Ensure migrations are applied
    with db.connect() as conn:
        MigrationRunner(conn).apply_pending()
    return db


@key_app.command("create")
def cmd_key_create(
    name: Annotated[str, typer.Option("--name", "-n", help="Name or label for the API key.")] = "default",
    db_path: Annotated[Path | None, typer.Option("--db", help="Path to SQLite database file.")] = None,
) -> None:
    """Generate a new API key. Plaintext is displayed once and never saved."""
    db = get_database(db_path)
    try:
        record, plaintext_key = create_key(db, name=name)
    except Exception as exc:
        console.print(f"[bold red]Error creating key:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("\n[bold green]✓ API Key Created Successfully[/bold green]\n")
    console.print(f"  [bold]Key ID:[/bold]     {record.key_id}")
    console.print(f"  [bold]Name:[/bold]       {record.name}")
    console.print(f"  [bold]API Key:[/bold]    [yellow]{plaintext_key}[/yellow]")
    console.print("\n[bold red]Important:[/bold red] Save this key now. It is never stored in plaintext and cannot be recovered.\n")


@key_app.command("list")
def cmd_key_list(
    db_path: Annotated[Path | None, typer.Option("--db", help="Path to SQLite database file.")] = None,
) -> None:
    """List all API keys. Hashes and secrets are never displayed."""
    db = get_database(db_path)
    keys = list_keys(db)

    if not keys:
        console.print("[dim]No API keys found.[/dim]")
        return

    table = Table(title="YourLLM API Keys", show_header=True, header_style="bold cyan")
    table.add_column("Key ID", style="bold")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Created At")
    table.add_column("Last Used At")

    for k in keys:
        status_color = "green" if k.status == "active" else "red"
        table.add_row(
            k.key_id,
            k.name,
            f"[{status_color}]{k.status}[/{status_color}]",
            k.created_at,
            k.last_used_at or "[dim]Never[/dim]",
        )

    console.print(table)


@key_app.command("revoke")
def cmd_key_revoke(
    key_id: Annotated[str, typer.Argument(help="ID of the key to revoke.")],
    db_path: Annotated[Path | None, typer.Option("--db", help="Path to SQLite database file.")] = None,
) -> None:
    """Revoke an active API key."""
    db = get_database(db_path)
    success = revoke_key(db, key_id=key_id)
    if success:
        console.print(f"[bold green]✓ Key '{key_id}' has been revoked.[/bold green]")
    else:
        console.print(f"[bold yellow]Key '{key_id}' was not found or was already revoked.[/bold yellow]")
        raise typer.Exit(code=1)


@key_app.command("rotate")
def cmd_key_rotate(
    key_id: Annotated[str, typer.Argument(help="ID of the key to rotate.")],
    db_path: Annotated[Path | None, typer.Option("--db", help="Path to SQLite database file.")] = None,
) -> None:
    """Revoke an active key and generate a new key replacement."""
    db = get_database(db_path)
    try:
        new_record, new_plaintext = rotate_key(db, key_id=key_id)
    except Exception as exc:
        console.print(f"[bold red]Error rotating key:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"\n[bold green]✓ Key '{key_id}' successfully rotated[/bold green]\n")
    console.print(f"  [bold]Old Key ID:[/bold]  {key_id} ([red]revoked[/red])")
    console.print(f"  [bold]New Key ID:[/bold]  {new_record.key_id}")
    console.print(f"  [bold]New Name:[/bold]    {new_record.name}")
    console.print(f"  [bold]New Key:[/bold]     [yellow]{new_plaintext}[/yellow]")
    console.print("\n[bold red]Important:[/bold red] Save this key now. It is never stored in plaintext and cannot be recovered.\n")


@app.command("serve")
def cmd_serve() -> None:
    """Start the YourLLM gateway and supervised inference engine."""
    from yourllm.main import run
    run()


if __name__ == "__main__":
    app()
