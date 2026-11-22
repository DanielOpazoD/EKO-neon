import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import { defineConfig, type Plugin } from 'vite';
import { stripGlslTemplateComments } from './src/simulator/renderer/gpu/glslStrip';

function glslStrip(): Plugin {
  return {
    name: 'echotwin-glsl-strip',
    apply: 'build',
    enforce: 'pre',
    transform(code, id) {
      if (
        !/[\\/]src[\\/]simulator[\\/]renderer[\\/]gpu[\\/]glsl(Common|Heart|Thorax|Passes|Image|Generated|VolumeHeart|Scatterers)\.ts$/.test(
          id,
        )
      ) {
        return null;
      }
      return { code: stripGlslTemplateComments(code), map: null };
    },
  };
}

export default defineConfig(() => {
  return {
    plugins: [
      react(),
      tailwindcss(),
      glslStrip(),
      // Copies src/assets into dist/ so root-relative URLs like
      // "/src/assets/images/..." still resolve in production builds. Do not remove.
      {
        name: 'copy-unbundled-src-assets',
        apply: 'build' as const,
        closeBundle(error?: Error): void {
          if (error) {
            return;
          }
          const distDir = path.resolve(__dirname, 'dist');
          const srcAssets = path.resolve(__dirname, 'src/assets');
          const distAssets = path.resolve(distDir, 'src/assets');
          if (fs.existsSync(distDir) && fs.existsSync(srcAssets)) {
            fs.cpSync(srcAssets, distAssets, { recursive: true, force: true });
          }
        },
      },
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    build: {
      target: 'es2022',
      sourcemap: false,
      chunkSizeWarningLimit: 1200,
    },
    worker: {
      format: 'es' as const,
      plugins: () => [glslStrip()],
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
