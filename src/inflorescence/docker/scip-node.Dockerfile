# scip-typescript and scip-python.
#
# Both are Node programs — scip-python is a fork of pyright, so indexing Python needs a
# JavaScript runtime and no Python at all. That is also why they share one image: the
# toolchain, not the target language, is what has to be installed.
#
# alpine is the right base here (Go and Node have no wheel problem); for a Python image it
# would be the wrong call, since musl forces source builds where manylinux wheels exist.
FROM node:22-alpine

# Versions pinned so that two builds of this file produce the same indexers — "build it
# yourself and compare" is meaningless against a moving latest.
RUN npm install -g --omit=dev \
      @sourcegraph/scip-typescript@0.4.0 \
      @sourcegraph/scip-python@0.6.6 \
 && npm cache clean --force

# scip-python shells out to `pip list` to enumerate the environment and to `git` for version
# metadata, and aborts before indexing anything if either is missing — so a Python indexer
# written in TypeScript still needs a Python interpreter next to it. Installed after the npm
# layer, which keeps that layer cache-shared and makes this an incremental ~60 MB.
#
# Consequence worth knowing: the pip it finds is the container's, not the project's, so
# third-party imports resolve as external symbols. Calls within the indexed project — the
# ones the graph is built from — resolve normally.
RUN apk add --no-cache python3 py3-pip git \
 && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# The repository is mounted read-only, so npm and friends must cache elsewhere.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/cache/xdg \
    npm_config_cache=/cache/npm

WORKDIR /repo

# A named volume is mounted here to keep toolchain caches across runs. The container runs
# as the invoking user, whose uid does not exist in the image, so the mount point must be
# world-writable — an empty named volume inherits this mode from the image.
RUN mkdir -p /cache && chmod 0777 /cache
