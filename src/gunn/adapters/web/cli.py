"""CLI for running the Web adapter server.

This module provides a command-line interface for starting the FastAPI-based
Web adapter server with configurable options.
"""

import json
import time

import click
import uvicorn
from pydantic import BaseModel

from gunn.adapters.web.server import AuthToken, create_web_adapter
from gunn.core.orchestrator import Orchestrator, OrchestratorConfig
from gunn.utils.telemetry import get_logger, setup_logging


class AuthTokenConfig(BaseModel):
    """Configuration for authentication tokens."""

    token: str
    world_id: str
    agent_id: str
    permissions: list[str]
    expires_at: float | None = None


class WebServerConfig(BaseModel):
    """Configuration for the web server."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    auth_tokens: list[AuthTokenConfig] = []


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file (JSON)",
)
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--workers", default=1, help="Number of worker processes")
@click.option("--log-level", default="info", help="Log level")
@click.option("--world-id", default="default", help="World ID")
@click.option(
    "--rate-limit-requests", default=100, help="Rate limit requests per window"
)
@click.option("--rate-limit-window", default=60, help="Rate limit window in seconds")
@click.option(
    "--auth-token",
    multiple=True,
    help="Auth token in format: token:world_id:agent_id:permission1,permission2",
)
def run_server(
    config: str | None,
    host: str,
    port: int,
    workers: int,
    log_level: str,
    world_id: str,
    rate_limit_requests: int,
    rate_limit_window: int,
    auth_token: tuple,
) -> None:
    """Run the Web adapter server."""
    # Setup logging
    setup_logging()
    logger = get_logger("gunn.web_cli")

    # Load configuration
    server_config = WebServerConfig()
    if config:
        with open(config) as f:
            config_data = json.load(f)
            server_config = WebServerConfig(**config_data)
    else:
        # Use CLI options
        server_config.host = host
        server_config.port = port
        server_config.workers = workers
        server_config.log_level = log_level
        server_config.rate_limit_requests = rate_limit_requests
        server_config.rate_limit_window = rate_limit_window

        # Parse auth tokens from CLI
        for token_str in auth_token:
            parts = token_str.split(":")
            if len(parts) >= 4:
                token_config = AuthTokenConfig(
                    token=parts[0],
                    world_id=parts[1],
                    agent_id=parts[2],
                    permissions=parts[3].split(",") if parts[3] else [],
                )
                server_config.auth_tokens.append(token_config)

    logger.info(
        "Starting web server",
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        world_id=world_id,
        auth_tokens_count=len(server_config.auth_tokens),
    )

    # Create orchestrator
    orchestrator_config = OrchestratorConfig()
    orchestrator = Orchestrator(orchestrator_config, world_id=world_id)

    # Prepare auth tokens
    auth_tokens: dict[str, AuthToken] = {}
    for token_config in server_config.auth_tokens:
        auth_tokens[token_config.token] = AuthToken(
            token=token_config.token,
            world_id=token_config.world_id,
            agent_id=token_config.agent_id,
            permissions=token_config.permissions,
            expires_at=token_config.expires_at,
        )

    # Create web adapter
    web_adapter = create_web_adapter(
        orchestrator=orchestrator,
        auth_tokens=auth_tokens,
        rate_limit_requests=server_config.rate_limit_requests,
        rate_limit_window=server_config.rate_limit_window,
    )

    # Run server
    uvicorn.run(
        web_adapter.app,
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        log_level=server_config.log_level,
    )


@click.command()
@click.option("--world-id", default="default", help="World ID")
@click.option("--agent-id", required=True, help="Agent ID")
@click.option(
    "--permissions",
    default="submit_intent,get_observation,stream_observations",
    help="Comma-separated permissions",
)
@click.option("--expires-in", type=int, help="Token expires in N seconds")
def generate_token(
    world_id: str, agent_id: str, permissions: str, expires_in: int | None
) -> None:
    """Generate an authentication token."""
    import secrets

    token = secrets.token_urlsafe(32)
    expires_at = None
    if expires_in:
        expires_at = time.time() + expires_in

    token_config = AuthTokenConfig(
        token=token,
        world_id=world_id,
        agent_id=agent_id,
        permissions=permissions.split(","),
        expires_at=expires_at,
    )

    click.echo("Generated authentication token:")
    click.echo(f"Token: {token}")
    click.echo(f"World ID: {world_id}")
    click.echo(f"Agent ID: {agent_id}")
    click.echo(f"Permissions: {permissions}")
    if expires_at:
        click.echo(f"Expires at: {expires_at} ({expires_in} seconds from now)")

    click.echo("\nCLI format:")
    click.echo(f"--auth-token {token}:{world_id}:{agent_id}:{permissions}")

    click.echo("\nJSON config format:")
    click.echo(json.dumps(token_config.dict(), indent=2))


@click.group()
def cli() -> None:
    """Web adapter CLI."""
    pass


cli.add_command(run_server, name="server")
cli.add_command(generate_token, name="generate-token")


if __name__ == "__main__":
    cli()
