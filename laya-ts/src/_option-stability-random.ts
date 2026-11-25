/** Python's integer-seeded random.Random, restricted to the shuffle operation used by
 * option stability. Keep it private: changing the RNG changes which probes run. */
export class StabilityRandom {
  private readonly state = new Uint32Array(624);
  private index = 624;

  constructor(seedKey: string) {
    let crc = 0xffffffff;
    for (const byte of new TextEncoder().encode(seedKey)) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    const seed = (crc ^ 0xffffffff) >>> 0;
    // CPython passes the integer's little-endian words to MT19937 init_by_array.
    // CRC32 fits in one word; init_genrand(seed) alone produces different shuffles.
    const mt = this.state;
    mt[0] = 19650218;
    for (let i = 1; i < mt.length; i++) {
      mt[i] = (Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1812433253) + i) >>> 0;
    }
    let i = 1;
    for (let k = 624; k > 0; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + seed) >>> 0;
      if (++i >= 624) { mt[0] = mt[623]; i = 1; }
    }
    for (let k = 623; k > 0; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0;
      if (++i >= 624) { mt[0] = mt[623]; i = 1; }
    }
    mt[0] = 0x80000000;
  }

  private next(): number {
    const mt = this.state;
    if (this.index >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        mt[i] = mt[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      }
      this.index = 0;
    }
    let y = mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  private below(n: number): number {
    const bits = 32 - Math.clz32(n);
    let r: number;
    // random._randbelow uses n.bit_length(), including when n is a power of two,
    // and consumes another word on rejection. A modulo would bias and desync it.
    do { r = this.next() >>> (32 - bits); } while (r >= n);
    return r;
  }

  shuffle<T>(values: T[]): void {
    for (let i = values.length - 1; i > 0; i--) {
      const j = this.below(i + 1);
      [values[i], values[j]] = [values[j], values[i]];
    }
  }
}
