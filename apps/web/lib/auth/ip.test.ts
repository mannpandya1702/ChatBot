import { describe, it, expect } from "vitest";
import { ipAllowed } from "./ip";

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
