/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Self-contained server bundle for the air-gapped Docker image
  // (deploy/airgap). No effect on `next dev` / Vercel.
  output: "standalone",
  // Pin the trace root to this package so the standalone server lands at
  // .next/standalone/server.js (not nested under apps/web/) regardless of a
  // parent monorepo lockfile — the Dockerfile depends on that layout.
  outputFileTracingRoot: import.meta.dirname,
};

export default nextConfig;
