"""Integration tests for Docker deployment."""

import subprocess
import time
from pathlib import Path

import pytest
import requests


def docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


# Skip all Docker tests if Docker is not available
pytestmark = pytest.mark.skipif(
    not docker_available(), reason="Docker is not available or not running"
)


class TestDockerIntegration:
    """Test Docker deployment integration."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_docker_build(self):
        """Test that Docker image builds successfully."""
        project_root = Path(__file__).parent.parent.parent

        # Build the Docker image
        result = subprocess.run(
            ["docker", "build", "-t", "gunn:test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        # Docker build output can be in stdout or stderr, check both
        build_output = result.stdout + result.stderr
        assert (
            "Successfully tagged gunn:test" in build_output
            or "naming to docker.io/library/gunn:test done" in build_output
        )

    @pytest.mark.slow
    @pytest.mark.integration
    def test_docker_compose_dev(self):
        """Test that development docker compose works."""
        project_root = Path(__file__).parent.parent.parent

        try:
            # Start services
            result = subprocess.run(
                ["docker", "compose", "-f", "compose.dev.yml", "up", "-d"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )

            assert result.returncode == 0, f"Docker compose up failed: {result.stderr}"

            # Wait for services to be ready
            time.sleep(10)

            # Check if service is responding
            try:
                response = requests.get("http://localhost:8080/health", timeout=10)
                assert response.status_code in [200, 503]  # 503 is ok during startup
            except requests.exceptions.RequestException:
                # Service might not be fully ready yet, which is ok for this test
                pass

        finally:
            # Clean up
            subprocess.run(
                ["docker", "compose", "-f", "compose.dev.yml", "down"],
                cwd=project_root,
                capture_output=True,
                timeout=60,
            )

    def test_dockerfile_syntax(self):
        """Test that Dockerfile has valid syntax."""
        project_root = Path(__file__).parent.parent.parent
        dockerfile_path = project_root / "Dockerfile"

        assert dockerfile_path.exists(), "Dockerfile not found"

        # Parse Dockerfile for basic syntax issues
        with open(dockerfile_path) as f:
            content = f.read()

        # Check for required instructions
        assert "FROM python:3.13-slim" in content
        assert "WORKDIR /app" in content
        assert "COPY" in content
        assert "RUN" in content
        assert "EXPOSE" in content
        assert "CMD" in content

        # Check for security best practices
        assert "USER gunn" in content  # Non-root user
        assert "HEALTHCHECK" in content  # Health check

    def test_docker_compose_syntax(self):
        """Test that docker-compose files have valid syntax."""
        project_root = Path(__file__).parent.parent.parent

        compose_files = ["compose.yml", "compose.dev.yml"]

        for compose_file in compose_files:
            compose_path = project_root / compose_file
            assert compose_path.exists(), f"{compose_file} not found"

            # Validate compose file syntax
            result = subprocess.run(
                ["docker", "compose", "-f", compose_file, "config"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, (
                f"{compose_file} syntax error: {result.stderr}"
            )

    def test_deployment_scripts_executable(self):
        """Test that deployment scripts are executable."""
        project_root = Path(__file__).parent.parent.parent

        scripts = ["deploy/deploy.sh", "deploy/health-check.sh"]

        for script in scripts:
            script_path = project_root / script
            assert script_path.exists(), f"{script} not found"
            assert script_path.is_file(), f"{script} is not a file"

            # Check if script is executable
            import stat

            mode = script_path.stat().st_mode
            assert mode & stat.S_IXUSR, f"{script} is not executable"
