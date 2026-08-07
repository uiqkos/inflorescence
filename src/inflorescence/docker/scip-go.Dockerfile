# scip-go, with the toolchain it drives.
#
# Shipping only the indexer binary does not work: scip-go loads packages through
# golang.org/x/tools/go/packages, which shells out to `go list` — so the Go toolchain has to
# be present at index time, not just at build time. That rules out the usual multi-stage
# "copy the binary into scratch" trick and sets the floor for this image.
#
# Kept separate from the Node image so that indexing a Python or TypeScript repository never
# pulls a Go toolchain it will not use.
# scip-go v0.2.7 declares `go >= 1.25`, and the toolchain refuses to build it on anything
# older; keep this base at or above that floor when bumping the indexer.
FROM golang:1.25-alpine

# git: the module loader fetches dependencies of the repository being indexed.
# The indexer version is pinned so that two builds of this file produce the same binary —
# "build it yourself and compare" is meaningless against a moving @latest.
RUN apk add --no-cache git ca-certificates \
 && go install github.com/scip-code/scip-go/cmd/scip-go@v0.2.7 \
 && rm -rf /go/pkg/mod /root/.cache/go-build

# The repository is mounted read-only, so every cache has to live somewhere writable. These
# are defaults; the caller passes the same values explicitly with -e.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/cache/xdg \
    GOMODCACHE=/cache/go/mod \
    GOCACHE=/cache/go/build

WORKDIR /repo

# A named volume is mounted here to keep toolchain caches across runs. The container runs
# as the invoking user, whose uid does not exist in the image, so the mount point must be
# world-writable — an empty named volume inherits this mode from the image.
RUN mkdir -p /cache && chmod 0777 /cache
