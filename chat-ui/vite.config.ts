import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` so the built bundle works from a subpath — GitHub Pages serves
// this repo under /sa-insights/, and absolute asset URLs would 404 there.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { port: 5173 },
  build: { outDir: "dist", sourcemap: false },
});
