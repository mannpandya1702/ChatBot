#!/usr/bin/env node
/**
 * Mint the Supabase `anon` and `service_role` API keys offline.
 *
 * Self-hosted Supabase signs these as HS256 JWTs with your project JWT secret.
 * The stock self-hosting bundle ships DEMO keys — you MUST replace them. This
 * script needs no network and no npm deps (Node's built-in crypto only).
 *
 *   node gen-jwt.mjs "<your-JWT_SECRET-at-least-32-chars>"
 *
 * Paste the printed ANON_KEY / SERVICE_ROLE_KEY into your .env. Keys are valid
 * for 10 years; regenerate (and rotate the secret) if either is exposed.
 */
import { createHmac } from "node:crypto";

const secret = process.argv[2];
if (!secret || secret.length < 32) {
  console.error("Usage: node gen-jwt.mjs <JWT_SECRET>  (secret must be >= 32 chars)");
  process.exit(1);
}

const b64url = (buf) =>
  Buffer.from(buf).toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

function sign(payload) {
  const header = { alg: "HS256", typ: "JWT" };
  const enc = `${b64url(JSON.stringify(header))}.${b64url(JSON.stringify(payload))}`;
  const sig = b64url(createHmac("sha256", secret).update(enc).digest());
  return `${enc}.${sig}`;
}

const iat = Math.floor(Date.now() / 1000);
const exp = iat + 60 * 60 * 24 * 365 * 10; // 10 years
const base = { iss: "supabase", iat, exp };

console.log("ANON_KEY=" + sign({ ...base, role: "anon" }));
console.log("SERVICE_ROLE_KEY=" + sign({ ...base, role: "service_role" }));
