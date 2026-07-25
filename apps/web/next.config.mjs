/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Self-contained server bundle for the Docker images (deploy/airgap,
  // deploy/hosted). Vercel builds its own serverless output and has no use for
  // a standalone server, so leave its build alone rather than have two output
  // modes racing to define the same thing.
  output: process.env.VERCEL ? undefined : "standalone",
  // Pin the trace root to this package so the standalone server lands at
  // .next/standalone/server.js (not nested under apps/web/) regardless of a
  // parent monorepo lockfile — the Dockerfile depends on that layout.
  outputFileTracingRoot: import.meta.dirname,
};

export default nextConfig;
