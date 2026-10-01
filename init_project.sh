#!/usr/bin/env bash
# =============================================================================
#  NeuroTrace AI — Project scaffolding script (Linux / macOS)
# -----------------------------------------------------------------------------
#  Creates the full directory tree with .gitkeep placeholders so empty folders
#  survive in Git. The data/ folder is created locally but is fully gitignored.
#
#  Usage:
#      bash init_project.sh
# =============================================================================
set -euo pipefail

# Always run from the folder that contains this script (the repo root)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  ================================================"
echo "   NeuroTrace AI — initializing project structure"
echo "  ================================================"
echo ""

DIRS=(
  # assets — logo, banner, diagrams
  "assets/banner"
  "assets/diagrams"
  "assets/logo"
  # data — GITIGNORED, local only
  "data/raw"
  "data/processed"
  "data/external"
  # docs — research papers, proposal, references, slides
  "docs/papers"
  "docs/proposal"
  "docs/references"
  "docs/slides"
  # ai_model — notebooks, training scripts, configs, artifacts
  "src/ai_model/notebooks"
  "src/ai_model/scripts"
  "src/ai_model/configs"
  "src/ai_model/checkpoints"
  "src/ai_model/exports"
  # backend — FastAPI app + Firebase Functions + tests
  "src/backend/app/api"
  "src/backend/app/core"
  "src/backend/app/models"
  "src/backend/app/schemas"
  "src/backend/app/services"
  "src/backend/functions"
  "src/backend/tests"
  # frontend — React web app
  "src/frontend/public"
  "src/frontend/src/components/canvas"
  "src/frontend/src/hooks"
  "src/frontend/src/pages"
  "src/frontend/src/services"
  "src/frontend/src/store"
  "src/frontend/src/utils"
  # mobile — Capacitor wrapper
  "src/mobile/android"
  "src/mobile/ios"
  "src/mobile/www"
)

# 1) Create every directory
for d in "${DIRS[@]}"; do
  mkdir -p "$d"
  echo "  [dir] $d"
done

# 2) Add .gitkeep placeholders (skip data/ — it is fully gitignored)
for d in "${DIRS[@]}"; do
  case "$d" in
    data*) continue ;;
  esac
  touch "$d/.gitkeep"
done

echo ""
echo "  Done! Directory tree:"
echo "  -----------------------------------------------"
if command -v tree >/dev/null 2>&1; then
  tree -a -I '.git|node_modules|.venv' --dirsfirst
else
  find . -path ./.git -prune -o -type d -print | sort | sed 's|^\./||' | sed '/^\.$/d'
fi
echo ""
echo "  Next steps:"
echo "    - Frontend : cd src/frontend && npm install && npm run dev"
echo "    - Backend  : cd src/backend  && python -m venv .venv && pip install -r requirements.txt"
echo "    - AI model : cd src/ai_model (Jupyter notebooks & training scripts)"
echo ""
