#!/usr/bin/env python3
"""
Launcher script for the battle demo frontend.

This script provides an easy way to start the Pygame-based battle visualization
with default settings and proper error handling.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the demo directory to the Python path
demo_dir = Path(__file__).parent
sys.path.insert(0, str(demo_dir))

from frontend.renderer import BattleRenderer


async def main():
    """Main function to run the battle renderer with default settings."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Default configuration
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    window_size = (800, 600)

    logger.info("Starting Battle Demo Frontend")
    logger.info(f"Backend URL: {backend_url}")
    logger.info(f"Window Size: {window_size[0]}x{window_size[1]}")
    logger.info("")
    logger.info("Controls:")
    logger.info("  SPACE - Pause/Resume game")
    logger.info("  ESC   - Quit application")
    logger.info("  D     - Toggle debug info")
    logger.info("  C     - Toggle communication display")
    logger.info("  R     - Reset game")
    logger.info("  F5    - Refresh game state")
    logger.info("")

    # Create and run renderer
    renderer = BattleRenderer(backend_url=backend_url, window_size=window_size)

    try:
        await renderer.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    finally:
        await renderer.cleanup()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
