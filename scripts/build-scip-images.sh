#!/usr/bin/env bash
# Build (and optionally push) the SCIP indexer images.
#
#   scripts/build-scip-images.sh                # build for the local platform
#   PUSH=1 scripts/build-scip-images.sh         # build both arches and push to the registry
#
# Two arches matter: users are on arm64 Macs and amd64 Linux in roughly equal measure, and a
# single-arch image fails at `docker run` time on the other half of them.
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/uiqkos}"
TAG="${TAG:-v1}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build() {
  local name="$1" dockerfile="$2"
  local image="${REGISTRY}/inflorescence-${name}:${TAG}"
  echo "==> ${image}"
  if [[ "${PUSH:-0}" == "1" ]]; then
    docker buildx build --platform "${PLATFORMS}" -t "${image}" -f "${dockerfile}" --push "${ROOT}/docker"
  else
    docker build -t "${image}" -f "${dockerfile}" "${ROOT}/docker"
  fi
  docker image inspect "${image}" --format '    size: {{.Size}} bytes' 2>/dev/null || true
}

build scip-go "${ROOT}/docker/scip-go.Dockerfile"
build scip-node "${ROOT}/docker/scip-node.Dockerfile"
