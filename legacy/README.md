# Legacy desktop app (archived)

The original PyQt5 version of Orpheus Negative Lab, kept for reference while the
web app in `../web` reaches feature parity. It is no longer maintained.

- `orpheus.py`: the app (rawpy + OpenCV + PyQt5)
- `app.ui`: an unused Qt Designer layout from an earlier version
- `setup.py`: the py2app build script

Why it was replaced: it decodes RAW files as 8-bit gamma-encoded images with
camera white balance and inverts them with `255 - x`. It then stretches each
channel linearly between endpoints picked from the histogram peak. The web app
works in linear float, divides out the measured film base, and inverts in
density space. See `../README.md`.
