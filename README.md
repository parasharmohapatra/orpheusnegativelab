# Orpheus Negative Lab

Convert colour negative, black-and-white and slide film scans into finished positives, entirely in your browser. Files never leave your device: decoding, conversion and export all run locally (WebAssembly, WebGL2, Web Workers).

## What it does

- **Opens RAW camera scans** (CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2, PEF, …) via LibRaw compiled to WebAssembly, plus 16-bit TIFFs from film scanners and JPEG/PNG.
- **Measures the orange mask.** It finds the film base on the rebate (or you sample it), then divides it out.
- **Inverts in density, like a darkroom print**, rather than with `255 − x`:
  - Each channel is normalized between robust shadow and highlight densities, which equalizes the dye layers and removes crossover.
  - Leftover midtone casts are measured on near-neutral pixels only, so colourful subjects aren't neutralized.
- **Roll-aware.** The film base is shared across the roll. Colour balance runs per frame (like NLP's auto), per roll (film response shared, exposure kept per frame), or a blend of the two.
- **Print look.** Paper-style curves (soft, standard, dense, linear), exposure, contrast, highlights/shadows/whites/blacks, white balance, vibrance/saturation, an 8-band colour mixer (OkLCh), and RGB and per-channel curves.
- **Film looks** inspired by Portra 160/400/800, Ektar, Gold, UltraMax, ColorPlus, Pro 400H, Superia, C200, Velvia, Provia, CineStill 800T/50D/400D, Vision3 250D/500T, plus B&W stocks (HP5, Tri-X, Delta, Pan F, T-Max, Fomapan).
  - Each look has a strength slider and a live preview on your own frame.
  - Looks layer on top of your adjustments rather than replacing them.
- **Library grid and bulk editing.** Select frames (click, ⇧ for a range, ⌘ to add), then use the selection bar to set the film type, balance or look, rotate, reset, sync or export them all at once.
  - With **Edit all** on, every slider in Develop adjusts the whole selection.
  - Undo reverts a bulk edit across all the frames it touched.
- **Crop, rotate and straighten**, with auto-frame from the detected film border. Cropping never changes the colour: density is always measured on the whole frame, with the holder and rebate excluded automatically.
- **Batch export** at full resolution: JPEG, or 8/16-bit TIFF and PNG, in sRGB, Display P3, Adobe RGB or ProPhoto, with an embedded ICC profile and an optional white border. Rendering is tiled on the GPU, so 50+ MP files work.
- **Remembers your edits** per file (IndexedDB), with presets, sync between frames, and undo/redo.

## Using it

1. Open a folder of scans, or drop files or a folder onto the page.
2. Each frame is converted automatically. Use **Convert** to:
   - choose the film type
   - sample the film base (`B`) or a neutral (`N`)
   - pick Frame, Roll or Blend balance
3. Refine with **Adjust**, **Colour** and **Curves**, and crop with **Crop** (`C`).
4. Press `G` for the **Library** grid. Select frames there, or in the filmstrip, to edit them together: the selection bar has film type, balance, looks, rotate, sync and export. Pick a film look under **Looks** (`P`).
5. **Export** (`E`):
   - Chromium browsers save into an `onl_positives/` folder you pick.
   - Other browsers download a file, or a ZIP for several frames.

Hold `\` to see the original negative. `←`/`→` move between frames.

## Development

```bash
cd web
npm install
npm run dev        # http://localhost:5173
npm test           # engine tests: synthetic film round trip, roll consistency, encoders
npm run build      # static site in web/dist (deployed to GitHub Pages by .github/workflows/pages.yml)
```

### Layout

| Path | Contents |
|---|---|
| `web/src/engine/` | Framework-free conversion engine |
| `web/src/engine/pipeline/` | Recipe schema, film-base and frame analysis, roll statistics, parameter building, and `reference.ts` (the CPU implementation that is the source of truth) |
| `web/src/engine/gpu/` | WebGL2 renderer; `convert.frag.glsl` mirrors `reference.ts` line for line |
| `web/src/engine/decode/`, `encode/`, `export/`, `workers/` | Decoders, ICC/TIFF/PNG/JPEG writers, tiled full-resolution export, worker pool |
| `web/src/app/` | React UI, zustand store, IndexedDB persistence, theme |
| `web/wasm/libraw/` | Reproducible LibRaw → WebAssembly build. The prebuilt module is committed in `web/public/wasm/`. |
| `legacy/` | The original PyQt desktop app, kept for reference |

For local testing with your own scans, the dev server can open files from disk:

```bash
ONL_DEV_FS=/path/to/scans npm run dev
# then open http://localhost:5173/?devload=/@fs/path/to/scans/a.CR2,/@fs/path/to/scans/b.CR2
```

This only works in development builds.

### How the conversion works

For each pixel with linear camera signal `I` and film base `B`:

1. Compute density above base: `D = −log10(I / B)`.
2. Normalize per channel between the 0.3% and 99.7% density percentiles inside the frame. Pixels brighter than the base and clipped pixels are excluded.
3. Apply a midtone bend fitted on near-grey pixels. It only adjusts midtones; shadows and highlights stay where step 2 put them.
4. Convert to scene light: `E = 0.18 · 2^(2.5 + R·(n − 1))`, where `R` is the scene range in stops.
5. Apply the camera matrix, exposure and white balance, giving linear Rec.2020.
6. Apply the paper sigmoid in log space, then colour adjustments in OkLCh.
7. Convert to the output colour space, encode, and apply curves.

The tests simulate a masked, crossed-over negative of a ColorChecker. The round trip recovers it with a mean ΔE2000 under 2, and grey stays neutral across a ±2 EV roll.

## License

MIT, see `LICENSE`. LibRaw is used under the LGPL 2.1 / CDDL 1.0 (see `web/wasm/libraw/README.md`).
