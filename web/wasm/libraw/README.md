# LibRaw → WebAssembly

Builds [LibRaw](https://www.libraw.org/) and a small C shim (`shim.c`) to
`web/public/wasm/libraw.{js,wasm}`, which `src/engine/decode/librawModule.ts`
loads lazily inside a Web Worker.

| | |
|---|---|
| LibRaw | **0.22.2** (`LIBRAW_VERSION`), commit `b93f6e45c194f5df9b02a43b1af9a54b4f41f33f` |
| Emscripten | **6.0.13** |
| Output | `libraw.wasm` 770 KB (302 KB gzip -9), `libraw.js` 12.5 KB |
| Threads | none: no pthreads, no SharedArrayBuffer, no COOP/COEP (works on GitHub Pages) |
| Exceptions | native wasm exceptions (`-fwasm-exceptions`; Chrome 95+, Firefox 100+, Safari 15.2+) |
| Optional deps | zlib on (Emscripten port; deflate/floating-point DNG). libjpeg, jasper and LCMS off |

Without libjpeg, **lossy (JPEG-compressed) DNGs** are rejected by LibRaw
(untested here: no sample on hand). Enabling them would need a libjpeg port
plus `-DUSE_JPEG -DUSE_JPEG8`. Lossless DNGs, CR2/CR3, NEF, ARW, RAF, ORF, RW2 and
the rest decode normally. JPEG-XL DNGs need Adobe's DNG SDK, which is never
built here.

## Reproducible build

```bash
# 1. emsdk, outside the repo. emsdk needs Python >= 3.10; if `python3` is older, set
#    EMSDK_PYTHON for both commands below, e.g. EMSDK_PYTHON=$HOME/.pyenv/versions/3.11.8/bin/python3
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
~/emsdk/emsdk install 6.0.13 && ~/emsdk/emsdk activate 6.0.13

# 2. build (sources ~/emsdk/emsdk_env.sh itself if emcc is not on PATH;
#    set EMSDK=/path/to/emsdk to use another checkout)
web/wasm/libraw/build.sh
```

`build.sh` shallow-clones the `LIBRAW_VERSION` tag into
`$BUILD_DIR` (default `~/.cache/orpheus-libraw-build`, outside the repo),
checks the commit hash, compiles the 79 sources listed in LibRaw's
`Makefile.dist` with `-O3 -fwasm-exceptions -DUSE_ZLIB`, and links them with
`shim.c` using:

```
-sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createLibRaw -sENVIRONMENT=worker,web
-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=64MB -sSTACK_SIZE=2MB -sFILESYSTEM=0
```

A clean rebuild (fresh clone, fresh build dir) is byte-identical:

```
07ccf1fe0dd278dd8c4b65d88b016a036506609d0f7760a8887158c246a9a9b5  libraw.js
2bef5e4710bcea8a24c206a3816cc23482412b249e11d455153a97b804a39f48  libraw.wasm
```

`build.sh --node-test` also writes `$BUILD_DIR/node/libraw.mjs`
(`-sENVIRONMENT=node`, otherwise the same) for testing decodes from Node.

## What the shim does

`orp_decode(buf, len, half_size, user_qual)` runs `libraw_open_buffer` →
`unpack` → `dcraw_process` → `dcraw_make_mem_image` with

- `output_bps=16`, `gamm={1,1}`, `no_auto_bright=1`
- `use_camera_wb=0`, `use_auto_wb=0`, `user_mul={1,1,1,1}`
- `output_color=0` (raw camera RGB), `highlight=0` (clip), `user_flip=0`
- `half_size`, `user_qual` from the caller

The output is linear, black-subtracted, 16-bit RGB with 65535 = sensor white.
The shim returns an `orp_result` struct (layout fixed by static asserts in
`shim.c` and mirrored in `librawModule.ts`). It holds:

- the pixels
- `flip` from the file, unapplied
- `rgb_cam`, `cam_xyz`, `pre_mul`, black and white level, read after `unpack`
  and before `dcraw_process` overwrites them
- the per-channel fraction of pixels ≥ 65500
- make, model, ISO, shutter, aperture, focal length and timestamp
- LibRaw's error string on failure

The processor is closed before `orp_decode` returns; `orp_free` releases the
pixels after JS copies them out.

## Verification (Canon EOS R8, 24 MP, Apple M-series)

- **Native LibRaw 0.22.2, same shim (clang):** bit-exact at half size. At full
  size with AHD, 15 of 73 M values differ (mean |Δ| 3e-5) with
  `-ffp-contract=off`; the rest is libm rounding inside AHD's CIELab step.
- **rawpy 0.24 (LibRaw 0.21.3):** a lossless DNG at half size is bit-exact
  (max |Δ| = 0). For CR3, 0.21.3 predates R8 support and crops the frame
  differently (5999×3999 vs 6022×4024), so the two can't be compared
  pixel-for-pixel.
- **Decode time in a browser Web Worker** (warm, first call in brackets):
  half-size 231 ms (387), full AHD 910 ms (1050). The WASM heap peaks at about
  410 MB for a full 24 MP decode and does not grow across repeated decodes.

## License

LibRaw is dual-licensed under **LGPL-2.1** or **CDDL-1.0**. Orpheus uses it
under **LGPL-2.1**:

- LibRaw is linked into a separate, replaceable `libraw.wasm`.
- The exact source version and build script are published here, so anyone can
  rebuild it and swap the file.

LibRaw's copyright notice and the LGPL-2.1 text must ship with the app, for
example on an about/licenses page. The LibRaw source tree contains
`LICENSE.LGPL` and `COPYRIGHT`.
