import { defineConfig } from 'vite';

// The Tauri shell serves the built frontend from src-tauri's frontendDist,
// which points at this directory's build output.
export default defineConfig({
  root: 'src',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'chrome110',
    sourcemap: false,
  },
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
});
