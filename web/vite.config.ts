import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Dev only: ONL_DEV_FS=/path/to/scans lets `?devload=/@fs/...` open files from disk for testing.
const devFs = process.env.ONL_DEV_FS;

// Served from https://<user>.github.io/orpheusnegativelab/ in production.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/orpheusnegativelab/' : '/',
  plugins: [react()],
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true },
  server: devFs ? { fs: { allow: ['.', devFs] } } : undefined,
  test: {
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
  },
}));
