#!/bin/bash
# Growing Up — Launch Web Dashboard
# Double-click this file to start the server

cd "$(dirname "$0")"
source venv/bin/activate
echo "Starting Growing Up at http://localhost:5001 ..."
open -a "Google Chrome" http://localhost:5001
python webapp/app.py
