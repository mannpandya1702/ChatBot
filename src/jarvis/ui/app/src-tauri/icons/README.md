# icons

`icon.png` and `icon.ico` are committed here because `npm run tauri build` fails outright
without them, and a bundle that cannot be produced is not much of a HUD.

Both are placeholders: an arc-reactor ring in the accent colour from `config.example.yaml`,
generated rather than drawn. Replace them with real artwork whenever you like, keeping the
same two file names and a 512x512 PNG.

Regenerate with:

```
uv run python scripts/make_icons.py
```

The script writes both files and needs nothing outside the standard library. `icon.ico`
carries 16, 32, 48, 64, 128, and 256 pixel entries, so Windows has a real bitmap at every
size it asks for instead of scaling one down.
