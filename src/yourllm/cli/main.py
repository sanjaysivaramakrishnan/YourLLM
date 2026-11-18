"""YourLLM CLI for local management inside the container."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from yourllm.auth.service import create_key, list_keys, revoke_key, rotate_key
from yourllm.cli.diagnostics import (
    DEFAULT_BENCH_PROMPT,
    BenchmarkSample,
    DiagnosticsError,
    run_benchmark,
    run_doctor,
)
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


@app.command("doctor")
def cmd_doctor(
    url: Annotated[
        str,
        typer.Option(
            "--url",
            "-u",
            envvar="YLLM_BASE_URL",
            help="YourLLM gateway base URL (e.g. http://127.0.0.1:8000).",
        ),
    ] = "http://127.0.0.1:8000",
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            envvar="YLLM_API_KEY",
            help="API key (or set YLLM_API_KEY environment variable).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model ID to test (auto-discovered from /v1/models if omitted).",
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="HTTP request timeout in seconds."),
    ] = 60.0,
    skip_tools: Annotated[
        bool,
        typer.Option("--skip-tools", help="Skip the streaming tool-calling compatibility probe."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON output."),
    ] = False,
) -> None:
    """Run live diagnostics: health, auth, model discovery, streaming, and Cline tool calling."""
    report = run_doctor(
        base_url=url,
        api_key=api_key,
        model=model,
        timeout_s=timeout,
        skip_tools=skip_tools,
    )

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
        if not report.all_passed:
            raise typer.Exit(code=1)
        return

    table = Table(
        title=f"YourLLM Doctor — {report.base_url}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Details")

    for check in report.checks:
        if not check.passed:
            status_cell = "[bold red]FAIL[/bold red]"
        elif check.warning:
            status_cell = "[bold yellow]WARN[/bold yellow]"
        else:
            status_cell = "[bold green]PASS[/bold green]"

        latency_cell = f"{check.latency_ms:.0f} ms" if check.latency_ms is not None else "[dim]-[/dim]"
        table.add_row(check.name, status_cell, latency_cell, check.detail)

    console.print()
    console.print(table)

    if report.all_passed:
        if report.warning_count > 0:
            console.print(
                f"\n[bold yellow]✓ Diagnostics completed with {report.warning_count} warning(s).[/bold yellow]\n"
            )
        else:
            console.print(
                "\n[bold green]✓ All diagnostics passed! Server is ready for coding agents.[/bold green]\n"
            )
    else:
        console.print("\n[bold red]✗ One or more diagnostic checks failed.[/bold red]\n")
        raise typer.Exit(code=1)


@app.command("bench")
def cmd_bench(
    url: Annotated[
        str,
        typer.Option(
            "--url",
            "-u",
            envvar="YLLM_BASE_URL",
            help="YourLLM gateway base URL (e.g. http://127.0.0.1:8000).",
        ),
    ] = "http://127.0.0.1:8000",
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            envvar="YLLM_API_KEY",
            help="API key (or set YLLM_API_KEY environment variable).",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model ID to benchmark (auto-discovered from /v1/models if omitted).",
        ),
    ] = None,
    runs: Annotated[
        int,
        typer.Option("--runs", "-n", min=1, max=50, help="Number of streaming benchmark iterations."),
    ] = 3,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", min=1, max=4096, help="Maximum completion tokens per run."),
    ] = 128,
    prompt: Annotated[
        str,
        typer.Option("--prompt", "-p", help="Prompt used for each benchmark iteration."),
    ] = DEFAULT_BENCH_PROMPT,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Per-request HTTP timeout in seconds."),
    ] = 120.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON output."),
    ] = False,
) -> None:
    """Benchmark live streaming TTFT (Time-To-First-Token) and throughput (tokens/sec)."""
    if not api_key:
        if json_output:
            typer.echo(json.dumps({"error": "API key is required. Pass --api-key or set YLLM_API_KEY."}))
        else:
            console.print(
                "[bold red]Error:[/bold red] API key is required. Pass [cyan]--api-key[/cyan] or set [cyan]YLLM_API_KEY[/cyan]."
            )
        raise typer.Exit(code=1)

    def _on_sample(sample: BenchmarkSample) -> None:
        if not json_output:
            est_note = " (est)" if sample.estimated_tokens else ""
            console.print(
                f"  [dim]Run {sample.run_index}/{runs}:[/dim] "
                f"TTFT [cyan]{sample.ttft_ms:.0f} ms[/cyan] | "
                f"[green]{sample.tokens_per_sec:.1f} tok/s[/green] | "
                f"{sample.completion_tokens}{est_note} tokens in {sample.total_s:.2f}s"
            )

    if not json_output:
        console.print(f"\n[bold]Running {runs} streaming benchmark run(s) against {url}...[/bold]\n")

    try:
        summary = run_benchmark(
            base_url=url,
            api_key=api_key,
            model=model,
            runs=runs,
            max_tokens=max_tokens,
            prompt=prompt,
            timeout_s=timeout,
            on_sample=_on_sample,
        )
    except DiagnosticsError as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}))
        else:
            console.print(f"\n[bold red]Benchmark failed:[/bold red] {exc}\n")
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(summary.to_dict(), indent=2))
        return

    table = Table(
        title=f"YourLLM Benchmark Summary — {summary.model}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Average", justify="right", style="bold green")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    table.add_row(
        "Time To First Token (TTFT)",
        f"{summary.avg_ttft_ms:.1f} ms",
        f"{summary.min_ttft_ms:.1f} ms",
        f"{summary.max_ttft_ms:.1f} ms",
    )
    table.add_row(
        "Throughput (Tokens/sec)",
        f"{summary.avg_tps:.1f} tok/s",
        f"{summary.min_tps:.1f} tok/s",
        f"{summary.max_tps:.1f} tok/s",
    )
    table.add_row(
        "End-to-End Latency",
        f"{summary.avg_total_s:.2f} s",
        f"{min(s.total_s for s in summary.samples):.2f} s",
        f"{max(s.total_s for s in summary.samples):.2f} s",
    )

    console.print()
    console.print(table)
    console.print(
        f"\n  [bold]Total Output Tokens:[/bold] {summary.total_completion_tokens} across {summary.runs} run(s)\n"
    )


@app.command("serve")
def cmd_serve() -> None:
    """Start the YourLLM gateway and supervised inference engine."""
    from yourllm.main import run
    run()


if __name__ == "__main__":
    app()
