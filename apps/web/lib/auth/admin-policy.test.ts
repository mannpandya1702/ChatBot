import { describe, expect, it } from "vitest";
import { assertAccessInput, canActOn } from "./admin-policy";

describe("canActOn — who may perform privileged admin operations", () => {
  it("lets a super_admin act on any role", () => {
    expect(canActOn("super_admin", "jawan")).toBe(true);
    expect(canActOn("super_admin", "admin")).toBe(true);
    expect(canActOn("super_admin", "super_admin")).toBe(true);
  });

  it("lets an admin act on jawans only", () => {
    expect(canActOn("admin", "jawan")).toBe(true);
    // An admin must not reset a peer's or a super_admin's credentials —
    // that would be a lateral/vertical privilege escalation.
    expect(canActOn("admin", "admin")).toBe(false);
    expect(canActOn("admin", "super_admin")).toBe(false);
  });

  it("denies everyone else, including unknown roles", () => {
    expect(canActOn("jawan", "jawan")).toBe(false);
    expect(canActOn("jawan", "admin")).toBe(false);
    expect(canActOn("", "jawan")).toBe(false);
    expect(canActOn("root", "jawan")).toBe(false);
  });
});

describe("assertAccessInput — role/tier change validation", () => {
  it("accepts every valid role and tier", () => {
    expect(() => assertAccessInput("jawan", 1)).not.toThrow();
    expect(() => assertAccessInput("admin", 2)).not.toThrow();
    expect(() => assertAccessInput("super_admin", 3)).not.toThrow();
  });

  it("accepts a partial change (role only, or tier only)", () => {
    expect(() => assertAccessInput("admin", undefined)).not.toThrow();
    expect(() => assertAccessInput(undefined, 2)).not.toThrow();
  });

  it("rejects a no-op change", () => {
    expect(() => assertAccessInput(undefined, undefined)).toThrow(/nothing to change/i);
  });

  it("rejects an unknown role", () => {
    expect(() => assertAccessInput("root", 1)).toThrow(/invalid role/i);
    expect(() => assertAccessInput("", 1)).toThrow(/invalid role/i);
  });

  it("rejects out-of-range and non-integer tiers", () => {
    expect(() => assertAccessInput(undefined, 0)).toThrow(/tier/i);
    expect(() => assertAccessInput(undefined, 4)).toThrow(/tier/i);
    expect(() => assertAccessInput(undefined, -1)).toThrow(/tier/i);
    expect(() => assertAccessInput(undefined, 1.5)).toThrow(/tier/i);
    expect(() => assertAccessInput(undefined, Number.NaN)).toThrow(/tier/i);
  });
});
