"""
Main entry point for the battle demo frontend.

This module provides the command-line interface for running the Pygame-based
battle visualization with configurable backend connection and display options.
"""

import asyncio
import logging
import sys

from .renderer import main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
