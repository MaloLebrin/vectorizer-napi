/**
 * Unit tests for vectorizer-napi addon.
 * Run: pnpm test (after pnpm build)
 * Uses Node.js built-in test runner (node --test).
 */
import { test, describe } from "node:test";
import assert from "node:assert";

// Dynamic import - addon must be built first (pnpm build)
const { modelName, vectorize } = await import("./index.js");

describe("modelName", () => {
  test("returns all-MiniLM-L6-v2", () => {
    assert.strictEqual(modelName(), "all-MiniLM-L6-v2");
  });

  test("returns a non-empty string", () => {
    const name = modelName();
    assert.strictEqual(typeof name, "string");
    assert.ok(name.length > 0);
  });
});

describe("vectorize", () => {
  test("returns a Promise", () => {
    const result = vectorize("hello");
    assert.ok(result instanceof Promise);
  });

  test("resolves to Float64Array of 384 dimensions", async () => {
    const vec = await vectorize("hello world");
    assert.ok(vec instanceof Float64Array, "should return Float64Array");
    assert.strictEqual(vec.length, 384, "all-MiniLM-L6-v2 produces 384-dim vectors");
  });

  test("produces non-zero embedding for non-empty text", async () => {
    const vec = await vectorize("test embedding");
    const hasNonZero = Array.from(vec).some((x) => x !== 0);
    assert.ok(hasNonZero, "embedding should have non-zero values");
  });

  test("same text produces same embedding", async () => {
    const vec1 = await vectorize("deterministic");
    const vec2 = await vectorize("deterministic");
    assert.deepStrictEqual(Array.from(vec1), Array.from(vec2));
  });
}).timeout(60_000); // Model loading can be slow on first run
