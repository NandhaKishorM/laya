import { describe, expect, it } from "vitest";
import { feed, feedHead } from "../src/providers.js";

const ort = {
  Tensor: class {
    type: string;
    data: any;
    dims: number[];
    constructor(type: string, data: any, dims: number[]) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  },
};

describe("flat parity", () => {
  it("feedHead flattens nested hidden without value drift", () => {
    const hidden = [
      [
        [1, 2],
        [3, 4],
      ],
    ];
    const batch: any = {
      inputIds: [[7, 8]],
      attentionMask: [[1, 1]],
      markerPos: [[0, 1]],
      markerMask: [[true, true]],
      qtype: [0],
    };
    const out = feedHead(ort, hidden, batch);
    expect(out.hidden_states.dims).toEqual([1, 2, 2]);
    expect(Array.from(out.hidden_states.data)).toEqual([1, 2, 3, 4]);
  });
  it("feed builds int64 ids+mask", () => {
    const out = feed(ort, {
      inputIds: [[5, 6]],
      attentionMask: [[1, 1]],
      markerPos: [[0]],
      markerMask: [[true]],
      qtype: [0],
    } as any);
    expect(out.input_ids.dims).toEqual([1, 2]);
    expect(Array.from(out.input_ids.data)).toEqual([5n, 6n]);
  });
});
