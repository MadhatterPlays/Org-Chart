#!/usr/bin/env bash
# Regenerate the dataset, retrain, re-score and rebuild web/index.html (takes a few minutes).
set -euo pipefail
cd "$(dirname "$0")/pipeline"
python3 generate.py
python3 infer.py
python3 evaluate.py
python3 build.py
echo "Done. Start the app with ./serve.sh"
