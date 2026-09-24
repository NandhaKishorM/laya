import { describe, expect, it } from "vitest";
import { maxOf } from "../src/common.js";

describe("maxOf", () => {
  it("returns max without spread overflow", () => {
    expect(maxOf([1, 5, 3], 1)).toBe(5);
    expect(maxOf([], 1)).toBe(1);
  });
  it("handles 300k rows without RangeError", () => {
    const lens = new Array(300_000).fill(1);
    lens[123456] = 77;
    expect(maxOf(lens, 1)).toBe(77);
  });
});
