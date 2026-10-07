#!/usr/bin/env bash
# Bump the shivvr version (Cargo.toml is the single source of truth) and
# refresh Cargo.lock to match. Edits files only: it does not commit, tag or push.
#
#   bash scripts/bump-version.sh 0.5.0      # explicit version
#   bash scripts/bump-version.sh minor      # 0.4.3 -> 0.5.0 (also: major, patch)
#
# Then: commit "release: v<version>", tag v<version>, push the tag. The tag push
# runs .github/workflows/release.yml, which fails if the tag and Cargo.toml disagree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <major|minor|patch|X.Y.Z>" >&2
    exit 2
fi

CURRENT="$(sed -nE 's/^version *= *"([^"]+)".*/\1/p' Cargo.toml | head -1)"
IFS=. read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$1" in
    major) NEXT="$((MAJOR + 1)).0.0" ;;
    minor) NEXT="${MAJOR}.$((MINOR + 1)).0" ;;
    patch) NEXT="${MAJOR}.${MINOR}.$((PATCH + 1))" ;;
    *)
        if [[ ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "not a version or bump kind: $1" >&2
            exit 2
        fi
        NEXT="$1"
        ;;
esac

# The new version must be ahead of the newest v* tag (sort -V handles 0.10 > 0.9).
LAST_TAG="$(git tag --list 'v*' --sort=-v:refname | head -1)"
LAST="${LAST_TAG#v}"
if [[ -n "$LAST" ]]; then
    HIGHEST="$(printf '%s\n%s\n' "$LAST" "$NEXT" | sort -V | tail -1)"
    if [[ "$NEXT" == "$LAST" || "$HIGHEST" != "$NEXT" ]]; then
        echo "version $NEXT is not ahead of the last tag $LAST_TAG" >&2
        exit 1
    fi
fi
if git rev-parse -q --verify "refs/tags/v${NEXT}" >/dev/null; then
    echo "tag v${NEXT} already exists" >&2
    exit 1
fi

# Only the [package] version line (the first `version = ` in the file).
awk -v next_version="$NEXT" '
    !done && /^version *= *"/ { print "version = \"" next_version "\""; done = 1; next }
    { print }
' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

# Refresh the shivvr entry in Cargo.lock without touching other dependencies,
# then make sure the release build still compiles.
cargo update --offline -p shivvr >/dev/null 2>&1 || cargo update -p shivvr
cargo check --release --locked

echo "==> ${CURRENT} -> ${NEXT}"
echo "Next:"
echo "  git commit -am \"release: v${NEXT}\""
echo "  git tag -a v${NEXT} -m \"shivvr v${NEXT}\""
echo "  git push origin main v${NEXT}"
