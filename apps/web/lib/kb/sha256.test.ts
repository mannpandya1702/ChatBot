import { describe, expect, it } from "vitest";
import { createHash, webcrypto } from "node:crypto";
import { sha256Hex, sha256HexOf } from "./sha256";

const bytes = (s: string) => new TextEncoder().encode(s);

describe("sha256Hex (WebCrypto-less fallback)", () => {
  it("matches the published NIST vectors", () => {
    expect(sha256Hex(bytes(""))).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    expect(sha256Hex(bytes("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(sha256Hex(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))).toBe(
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    );
  });

  it("pads correctly at every length across the two-block boundary", () => {
    // 55/56 and 119/120 are where the length field stops fitting and a second
    // padding block is required — the classic off-by-one in a hand-rolled SHA.
    for (const n of [0, 1, 54, 55, 56, 57, 63, 64, 65, 118, 119, 120, 121, 127, 128, 129, 1000]) {
      const input = new Uint8Array(n).map((_, i) => (i * 31 + 7) & 0xff);
      expect(sha256Hex(input), `length ${n}`).toBe(
        createHash("sha256").update(input).digest("hex"),
      );
    }
  });

  it("hashes a multi-megabyte buffer identically to node crypto", () => {
    const big = new Uint8Array(3 * 1024 * 1024 + 137);
    for (let i = 0; i < big.length; i++) big[i] = (i * 2654435761) & 0xff;
    expect(sha256Hex(big)).toBe(createHash("sha256").update(big).digest("hex"));
  });

  it("reads a view into a larger buffer, not the whole buffer", () => {
    const backing = new Uint8Array(256).map((_, i) => i & 0xff);
    const slice = backing.subarray(64, 96);
    expect(sha256Hex(slice)).toBe(createHash("sha256").update(slice).digest("hex"));
  });
});

describe("sha256HexOf", () => {
  const input = bytes("सैनिक सहायक");
  const expected = createHash("sha256").update(input).digest("hex");

  it("uses WebCrypto when a secure context provides it", async () => {
    const original = globalThis.crypto;
    Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
    try {
      expect(await sha256HexOf(input)).toBe(expected);
    } finally {
      Object.defineProperty(globalThis, "crypto", { value: original, configurable: true });
    }
  });

  it("falls back to the JS implementation when crypto.subtle is absent", async () => {
    const original = globalThis.crypto;
    Object.defineProperty(globalThis, "crypto", { value: undefined, configurable: true });
    try {
      expect(await sha256HexOf(input)).toBe(expected);
    } finally {
      Object.defineProperty(globalThis, "crypto", { value: original, configurable: true });
    }
  });
});
