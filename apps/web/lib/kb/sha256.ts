/**
 * SHA-256 over bytes, in plain JS.
 *
 * The browser normally does this via `crypto.subtle`, but WebCrypto is only
 * exposed in a *secure context* — HTTPS or localhost. An air-gapped install
 * reached over the unit LAN (`http://10.0.0.5:3000`) is neither, and there
 * `crypto.subtle` is simply `undefined`. Since the upload path needs the
 * document's checksum before it sends anything, no fallback would mean no
 * uploads at all on exactly the deployment that matters most.
 *
 * Verified against the NIST vectors in sha256.test.ts.
 */

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

// Reused across blocks — a 50 MB file is ~800k compressions, and allocating a
// message schedule for each one is the difference between fast and unusable.
const W = new Uint32Array(64);

function compress(H: Uint32Array, view: DataView, offset: number): void {
  for (let t = 0; t < 16; t++) W[t] = view.getUint32(offset + t * 4, false);
  for (let t = 16; t < 64; t++) {
    const x = W[t - 15];
    const y = W[t - 2];
    const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
    const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
    W[t] = (W[t - 16] + s0 + W[t - 7] + s1) >>> 0;
  }

  let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
  for (let t = 0; t < 64; t++) {
    const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
    const ch = (e & f) ^ (~e & g);
    const t1 = (h + S1 + ch + K[t] + W[t]) >>> 0;
    const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
    const maj = (a & b) ^ (a & c) ^ (b & c);
    const t2 = (S0 + maj) >>> 0;
    h = g; g = f; f = e; e = (d + t1) >>> 0;
    d = c; c = b; b = a; a = (t1 + t2) >>> 0;
  }

  H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0;
  H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
  H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0;
  H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
}

/** Lowercase hex SHA-256 digest of `bytes`. */
export function sha256Hex(bytes: Uint8Array): string {
  const H = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);

  const len = bytes.length;
  const full = len - (len % 64);
  // Hash whole blocks straight out of the input — no padded copy, which for a
  // 50 MB PDF would otherwise double peak memory.
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let off = 0; off < full; off += 64) compress(H, view, off);

  // Tail: remaining bytes + 0x80 + zero padding + 64-bit big-endian bit length.
  // Needs two blocks when the remainder leaves no room for the length field.
  const rest = len - full;
  const tail = new Uint8Array(rest + 9 <= 64 ? 64 : 128);
  tail.set(bytes.subarray(full));
  tail[rest] = 0x80;
  const tailView = new DataView(tail.buffer);
  const bitLen = len * 8; // exact in a double for any file below 1 PB
  tailView.setUint32(tail.length - 8, Math.floor(bitLen / 0x100000000), false);
  tailView.setUint32(tail.length - 4, bitLen % 0x100000000, false);
  for (let off = 0; off < tail.length; off += 64) compress(H, tailView, off);

  let hex = "";
  for (let i = 0; i < 8; i++) hex += H[i].toString(16).padStart(8, "0");
  return hex;
}

/** WebCrypto when it exists (secure context), the JS implementation otherwise. */
export async function sha256HexOf(bytes: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const digest = await subtle.digest("SHA-256", bytes as unknown as ArrayBuffer);
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  }
  return sha256Hex(bytes);
}
