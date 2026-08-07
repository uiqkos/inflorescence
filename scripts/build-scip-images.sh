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
# The Dockerfiles live inside the package so that an installed wheel can rebuild the
# images too (`inflorescence build-images`); this script is the repo-checkout equivalent.
DOCKER_DIR="${ROOT}/src/inflorescence/docker"

build() {
  local name="$1" dockerfile="$2"
  local image="${REGISTRY}/inflorescence-${name}:${TAG}"
  echo "==> ${image}"
  if [[ "${PUSH:-0}" == "1" ]]; then
    docker buildx build --platform "${PLATFORMS}" -t "${image}" -f "${dockerfile}" --push "${DOCKER_DIR}"
  else
    docker build -t "${image}" -f "${dockerfile}" "${DOCKER_DIR}"
  fi
  docker image inspect "${image}" --format '    size: {{.Size}} bytes' 2>/dev/null || true
}

build scip-go "${DOCKER_DIR}/scip-go.Dockerfile"
build scip-node "${DOCKER_DIR}/scip-node.Dockerfile"
