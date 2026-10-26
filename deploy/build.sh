#!/usr/bin/env bash
# Build step (Render runs this; it also works locally on Linux/macOS or Git Bash).
#   - installs CPU-only PyTorch (no 3 GB of GPU libraries on a CPU server)
#   - downloads DINOv2 into $HF_HOME so the first request doesn't wait for it
#   - with PETMATCH_DEMO=1: fetches 300 DogFaceNet dogs and builds seed/demo.db
set -euo pipefail
cd "$(dirname "$0")/.."
python --version

pip install --upgrade pip
if [ "${SKIP_TORCH_INSTALL:-0}" != "1" ]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt

echo "Downloading the DINOv2 model..."
python -c "from petmatch.embedders import DinoV2Embedder; DinoV2Embedder()"

if [ "${PETMATCH_DEMO:-0}" = "1" ]; then
  echo "Building the demo database..."
  python scripts/fetch_dogfacenet.py --max-dogs 300 --workers 8
  mkdir -p seed
  rm -f seed/demo.db seed/demo.db.found.txt
  python scripts/seed_demo.py --dogs 300 --db seed/demo.db
fi
echo "Build done."
