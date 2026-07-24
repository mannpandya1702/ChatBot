import { describe, it, expect } from "vitest";
import { ipAllowed, clientIpFromHeaders } from "./ip";

describe("ipAllowed", () => {
  it("allows everything when the allowlist is empty (disabled)", () => {
    expect(ipAllowed("8.8.8.8", [])).toBe(true);
    expect(ipAllowed(null, [])).toBe(true);
  });

  it("matches within a CIDR range and rejects outside it", () => {
    expect(ipAllowed("10.1.2.3", ["10.0.0.0/8"])).toBe(true);
    expect(ipAllowed("11.1.2.3", ["10.0.0.0/8"])).toBe(false);
    expect(ipAllowed("192.168.1.50", ["192.168.1.0/24"])).toBe(true);
    expect(ipAllowed("192.168.2.50", ["192.168.1.0/24"])).toBe(false);
  });

  it("supports a bare host (/32 implied) and multiple entries", () => {
    expect(ipAllowed("203.0.113.7", ["203.0.113.7"])).toBe(true);
    expect(ipAllowed("203.0.113.8", ["203.0.113.7"])).toBe(false);
    expect(ipAllowed("172.16.5.5", ["10.0.0.0/8", "172.16.0.0/12"])).toBe(true);
  });

  it("fails closed when enabled but the ip is unknown", () => {
    expect(ipAllowed(null, ["10.0.0.0/8"])).toBe(false);
  });

  it("rejects malformed input", () => {
    expect(ipAllowed("not.an.ip", ["10.0.0.0/8"])).toBe(false);
    expect(ipAllowed("10.0.0.1", ["bad/cidr"])).toBe(false);
  });
});

describe("clientIpFromHeaders (X-Forwarded-For spoof resistance)", () => {
  it("falls back to x-real-ip when there is no XFF", () => {
    expect(clientIpFromHeaders(null, "9.9.9.9")).toBe("9.9.9.9");
    expect(clientIpFromHeaders(null, null)).toBe(null);
  });

  it("takes the rightmost (trusted) entry by default, not the client-controlled leftmost", () => {
    // Attacker prepends a fake allowlisted IP; the proxy appends the real one.
    expect(clientIpFromHeaders("10.0.0.1, 203.0.113.9", null)).toBe("203.0.113.9");
    expect(clientIpFromHeaders("203.0.113.9", null)).toBe("203.0.113.9");
  });

  it("honours trustedProxyCount for multi-proxy edges", () => {
    // client, cdn, our-nginx  → with 1 trusted proxy (nginx), the client is 'cdn'... no:
    // parts=[client, cdn]; count=1 → index len-1-1 = 0 → client
    expect(clientIpFromHeaders("198.51.100.7, 10.0.0.2", null, 1)).toBe("198.51.100.7");
    // over-counting clamps to the leftmost rather than going negative
    expect(clientIpFromHeaders("198.51.100.7, 10.0.0.2", null, 9)).toBe("198.51.100.7");
  });

  it("trims whitespace and ignores empty segments", () => {
    expect(clientIpFromHeaders("  10.0.0.1 , 203.0.113.9 ", null)).toBe("203.0.113.9");
    expect(clientIpFromHeaders(",,", null)).toBe(null);
  });
});
