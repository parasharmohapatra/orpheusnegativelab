#!/usr/bin/env bash
# Build LibRaw + shim.c to WebAssembly for Orpheus Negative Lab.
#
#   web/wasm/libraw/build.sh              -> web/public/wasm/libraw.{js,wasm}
#   web/wasm/libraw/build.sh --node-test  -> also $BUILD_DIR/node/libraw.mjs (Node build, for testing only)
#
# Needs an activated emsdk (EMSDK_VERSION below). If `emcc` is not on PATH the
# script sources $EMSDK/emsdk_env.sh (default ~/emsdk). See README.md.
set -euo pipefail

EMSDK_VERSION=6.0.13
LIBRAW_REPO=https://github.com/LibRaw/LibRaw.git
LIBRAW_COMMIT=b93f6e45c194f5df9b02a43b1af9a54b4f41f33f # tag 0.22.2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB="$(cd "$HERE/../.." && pwd)"
LIBRAW_TAG="$(tr -d '[:space:]' < "$HERE/LIBRAW_VERSION")"
BUILD_DIR="${BUILD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/orpheus-libraw-build}"
OUT_DIR="$WEB/public/wasm"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
NODE_TEST=0
[[ "${1:-}" == "--node-test" ]] && NODE_TEST=1

# --- toolchain -------------------------------------------------------------
if ! command -v emcc >/dev/null 2>&1; then
  EMSDK="${EMSDK:-$HOME/emsdk}"
  # shellcheck disable=SC1091
  source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1
fi
have="$(emcc --version | head -1 | sed -E 's/.* ([0-9]+\.[0-9]+\.[0-9]+) .*/\1/')"
if [[ "$have" != "$EMSDK_VERSION" ]]; then
  echo "warning: emcc $have, expected $EMSDK_VERSION (output may differ)" >&2
fi

# --- source ----------------------------------------------------------------
SRC="$BUILD_DIR/src"
if [[ ! -d "$SRC/.git" ]]; then
  mkdir -p "$BUILD_DIR"
  git clone --quiet --depth 1 --branch "$LIBRAW_TAG" "$LIBRAW_REPO" "$SRC"
fi
got="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$got" != "$LIBRAW_COMMIT" ]]; then
  echo "error: $SRC is at $got, expected $LIBRAW_COMMIT ($LIBRAW_TAG)" >&2
  exit 1
fi

# --- compile LibRaw (same object list as Makefile.dist) ---------------------
COMMON=(-O3 -fwasm-exceptions -DUSE_ZLIB -sUSE_ZLIB=1 -DLIBRAW_NODLL)
OBJ="$BUILD_DIR/obj"
mkdir -p "$OBJ"
SOURCES=()
while IFS= read -r name; do
  f="$(find "$SRC/src" -name "$name.cpp" | head -1)"
  [[ -n "$f" ]] || { echo "error: no source for $name" >&2; exit 1; }
  SOURCES+=("$f")
done < <(sed -n '/^LIB_OBJECTS/,/^$/p' "$SRC/Makefile.dist" | grep -o 'object/[a-z0-9_]*\.o' | sed 's#object/##;s#\.o$##')

echo "compiling ${#SOURCES[@]} LibRaw sources with $JOBS jobs..."
export SRC OBJ
printf '%s\n' "${SOURCES[@]}" | xargs -P "$JOBS" -n 1 sh -c '
  o="$OBJ/$(basename "$1" .cpp).o"
  [ "$o" -nt "$1" ] && exit 0
  exec em++ '"${COMMON[*]}"' -std=c++14 -w -I"$SRC" -c "$1" -o "$o"
' sh
rm -f "$BUILD_DIR/libraw.a"
emar rcs "$BUILD_DIR/libraw.a" "$OBJ"/*.o

# --- shim + link -----------------------------------------------------------
emcc "${COMMON[@]}" -std=c11 -Wall -I"$SRC" -c "$HERE/shim.c" -o "$BUILD_DIR/shim.o"
LINK=(
  "${COMMON[@]}" "$BUILD_DIR/shim.o" "$BUILD_DIR/libraw.a"
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createLibRaw
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=64MB
  -sSTACK_SIZE=2MB -sFILESYSTEM=0
  -sEXPORTED_FUNCTIONS=_orp_decode,_orp_free,_orp_version,_malloc,_free
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64,UTF8ToString
)

mkdir -p "$OUT_DIR"
em++ "${LINK[@]}" -sENVIRONMENT=worker,web -o "$OUT_DIR/libraw.js"
echo "wrote $OUT_DIR/libraw.js, $OUT_DIR/libraw.wasm"

if [[ $NODE_TEST == 1 ]]; then
  mkdir -p "$BUILD_DIR/node"
  em++ "${LINK[@]}" -sENVIRONMENT=node -o "$BUILD_DIR/node/libraw.mjs"
  echo "wrote $BUILD_DIR/node/libraw.mjs"
fi
