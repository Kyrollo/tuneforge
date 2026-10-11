#!/usr/bin/env bash
set -euo pipefail

echo "==> Running smoke check against http://localhost:8011..."
curl -fsS "http://localhost:8011/healthz"
echo ""
echo "==> Smoke check passed!"
