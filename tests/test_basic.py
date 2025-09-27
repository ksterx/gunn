"""Basic tests to verify project setup."""

import pytest

from gunn import __version__


def test_version() -> None:
    """Test that version is accessible."""
    assert __version__ == "0.1.0"


def test_import() -> None:
    """Test that the package can be imported."""
    import gunn

    assert hasattr(gunn, "__version__")


@pytest.mark.asyncio
async def test_async_support() -> None:
    """Test that async/await works in the test environment."""

    async def dummy_async() -> str:
        return "async works"

    result = await dummy_async()
    assert result == "async works"
