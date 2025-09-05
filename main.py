#!/usr/bin/env python3
"""
Main entry point for the PresBot Server application.
This file serves as the primary entry point for running the application.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6969, debug=True)
