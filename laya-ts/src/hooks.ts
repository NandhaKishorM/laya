/**
 * Opt-in prediction hooks: observe or shape every decision without forking.
 *
 * A hook is either a plain `(ctx) => void` or an object implementing any subset of the
 * lifecycle methods on `Hook`. Hooks are configured on `Agent` / `Router` and can be
 * overridden per call. Port of `laya/hooks.py`.
 */

import { pyRepr } from "./common.js";

/** Mutable state passed to every hook for one call. */
export class PredictContext {
  /** `states` / `questions` may be rewritten by `onPredictStart`. */
  states: unknown[];
  questions: Record<string, unknown>;
  /** Shared by every hook of one call. */
  runId: string;
  /** Set by `skip()` or after inference; `onPredictEnd` may rewrite it. */
  results: Record<string, unknown>[] | null = null;
  /** Router: the RouteDecision. */
  decision: Record<string, unknown> | null = null;
  /** Resolved checkpoint name. */
  model: string | null = null;
  agent: unknown = null;
  router: unknown = null;
  /** Per-call token-budget overrides; null = agent config. */
  maxLen: number | null = null;
  headMaxLen: number | null = null;
  /** Aggregated input/output tokens across results. */
  usage: Record<string, number> | null = null;
  startedAt: number;
  elapsedMs: number | null = null;
  error: unknown = null;

  constructor(init: { states: unknown[]; questions: Record<string, unknown> } & Partial<PredictContext>) {
    this.states = init.states;
    this.questions = init.questions;
    this.runId = newRunId();
    this.startedAt = now();
    for (const [k, v] of Object.entries(init)) {
      if (k !== "states" && k !== "questions" && v !== undefined) {
        (this as unknown as Record<string, unknown>)[k] = v;
      }
    }
  }

  /** Set cached results from a start hook; inference is skipped, end hooks still run. */
  skip(results: Record<string, unknown>[]): void {
    this.results = results;
  }

  /** Set elapsedMs from startedAt. Called by the dispatching call, not by hooks. */
  markElapsed(): void {
    this.elapsedMs = now() - this.startedAt;
  }
}

function newRunId(): string {
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID().replaceAll("-", "");
  let hex = "";
  for (let i = 0; i < 32; i++) hex += Math.floor(Math.random() * 16).toString(16);
  return hex;
}

function now(): number {
  const p = (globalThis as { performance?: { now?: () => number } }).performance;
  return p && typeof p.now === "function" ? p.now() : Date.now();
}

/** Optional lifecycle methods. Implement any subset; missing methods are skipped. */
export interface Hook {
  onPredictStart?(ctx: PredictContext): void | Promise<void>;
  onPredictEnd?(ctx: PredictContext): void | Promise<void>;
  onRoute?(ctx: PredictContext): void;
  onLoad?(ctx: PredictContext): void | Promise<void>;
  onEvict?(ctx: PredictContext): void | Promise<void>;
  onError?(ctx: PredictContext): void | Promise<void>;
}

export type PredictHook = (ctx: PredictContext) => void | Promise<void>;
export type HookArg = Hook | Hook[] | null | undefined;
export type PredictHookArg = PredictHook | PredictHook[] | null | undefined;

export const HOOK_EVENTS = [
  "onPredictStart",
  "onPredictEnd",
  "onRoute",
  "onLoad",
  "onEvict",
  "onError",
] as const;
export type HookEvent = (typeof HOOK_EVENTS)[number];

/**
 * No-op base class: subclass it and override only the events you need.
 *
 * `Hook` is the structural interface; `BaseHook` is the concrete convenience when you would
 * rather subclass than implement methods by shape. Every method does nothing by default.
 */
export class BaseHook implements Hook {
  onPredictStart(_ctx: PredictContext): void {}
  onPredictEnd(_ctx: PredictContext): void {}
  onRoute(_ctx: PredictContext): void {}
  onLoad(_ctx: PredictContext): void {}
  onEvict(_ctx: PredictContext): void {}
  onError(_ctx: PredictContext): void {}
}

function isHookLike(item: unknown): item is Hook {
  return (
    typeof item === "object" &&
    item !== null &&
    HOOK_EVENTS.some((e) => typeof (item as Record<string, unknown>)[e] === "function")
  );
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "Array";
  if (typeof v === "object") return (v as object).constructor?.name ?? "object";
  return typeof v;
}

/**
 * Normalise the hook arguments shared by Agent, Router and every predict call: an optional
 * hook (object or list) plus optional plain start/end callables. A plain callable is not a
 * hook on its own -- which lifecycle event would it bind to? Passing one as `hooks` is a
 * TypeError; pass it as `onPredictStart` / `onPredictEnd` instead. The same callable as both
 * start and end is a TypeError too: the registry dedupes by identity, so one of the two would
 * silently never fire.
 */
export function normaliseHooks(
  hooks?: unknown,
  onPredictStart?: PredictHookArg,
  onPredictEnd?: PredictHookArg,
): Hook[] {
  const out: Hook[] = [];
  const append = (item: unknown): void => {
    if (item === null || item === undefined) return;
    if (typeof item === "function") {
      // A JS function can carry properties; one that also defines hook methods is ambiguous.
      if (HOOK_EVENTS.some((e) => typeof (item as unknown as Record<string, unknown>)[e] === "function")) {
        throw new TypeError(
          "a hook must be a plain callable or implement hook methods, not both",
        );
      }
      throw new TypeError(
        "a plain callable is ambiguous; pass it as onPredictStart or onPredictEnd",
      );
    }
    if (Array.isArray(item)) {
      for (const h of item) append(h);
      return;
    }
    if (typeof item !== "object") {
      throw new TypeError(`hooks must be Hook objects, got ${typeName(item)}`);
    }
    if (!isHookLike(item)) {
      throw new TypeError(
        `a hook must implement at least one hook method (${HOOK_EVENTS.join(", ")})`,
      );
    }
    out.push(item as Hook);
  };
  append(hooks);
  if (onPredictStart !== null && onPredictStart !== undefined) {
    if (typeof onPredictStart !== "function") {
      throw new TypeError(`onPredictStart must be callable, got ${typeName(onPredictStart)}`);
    }
    out.push({ onPredictStart });
  }
  if (onPredictEnd !== null && onPredictEnd !== undefined) {
    if (typeof onPredictEnd !== "function") {
      throw new TypeError(`onPredictEnd must be callable, got ${typeName(onPredictEnd)}`);
    }
    out.push({ onPredictEnd });
  }
  if (
    onPredictStart !== null &&
    onPredictStart !== undefined &&
    onPredictStart === (onPredictEnd as unknown)
  ) {
    throw new TypeError(
      "onPredictStart and onPredictEnd must be different callables; one callable cannot serve both events",
    );
  }
  return out;
}

const defaultHooksList: Hook[] = [];

/** The process-wide hooks, a copy, in order. Empty unless set via `setDefaultHooks`. */
export function defaultHooks(): Hook[] {
  return [...defaultHooksList];
}

/**
 * Replace the process-wide default hooks.
 *
 * Defaults run before installed and per-call hooks for every `Agent` and `Router` in the
 * process, so a tracer or metrics hook does not have to be threaded through every
 * construction. Accepts the same arguments as the `hooks` option.
 */
export function setDefaultHooks(
  hooks?: HookArg,
  onPredictStart?: PredictHookArg,
  onPredictEnd?: PredictHookArg,
): void {
  const normalised = normaliseHooks(hooks, onPredictStart, onPredictEnd);
  defaultHooksList.length = 0;
  defaultHooksList.push(...normalised);
}

/** Append one hook or a list of hooks to the process-wide defaults. */
export function addDefaultHook(hook: HookArg): void {
  defaultHooksList.push(...normaliseHooks(hook));
}

/** Remove every process-wide default hook. */
export function clearDefaultHooks(): void {
  defaultHooksList.length = 0;
}

const defaultsRan = new WeakSet<object>();

/** `Router.predict` marks the options of its Agent call: the Router already ran the defaults for this request. */
export function markDefaultsRan(opts: object): void {
  defaultsRan.add(opts);
}

/** Whether `opts` was marked by `markDefaultsRan`, so the Agent must not run the defaults a second time. */
export function defaultsAlreadyRan(opts: object): boolean {
  return defaultsRan.has(opts);
}

/**
 * Effective hook list for one call: defaults, then installed, then per-call hooks.
 *
 * Reads the process-wide defaults at call time, so hooks set after construction still apply.
 */
export function composeHooks(
  installed: readonly Hook[],
  hooks?: HookArg,
  onPredictStart?: PredictHookArg,
  onPredictEnd?: PredictHookArg,
): Hook[] {
  if (
    defaultHooksList.length === 0 && installed.length === 0 && (hooks === null || hooks === undefined) &&
    (onPredictStart === null || onPredictStart === undefined) &&
    (onPredictEnd === null || onPredictEnd === undefined)
  ) {
    return EMPTY_HOOKS;
  }
  return [...defaultHooksList, ...installed, ...normaliseHooks(hooks, onPredictStart, onPredictEnd)];
}

const EMPTY_HOOKS: Hook[] = [];
Object.freeze(EMPTY_HOOKS);

/**
 * Base class giving a runtime-mutable hook list.
 *
 * `Agent` and `Router` extend it so hooks can be added, removed or scoped after construction.
 * Mutation replaces the array, so a call that already snapshotted its active hooks is never
 * disturbed by an add or remove that lands mid-flight.
 */
export class HookRegistry {
  hooks: Hook[] = [];

  /** Install one hook or a list of them. Returns this for chaining. */
  addHook(hook: HookArg): this {
    const added = normaliseHooks(hook);
    if (added.length > 0) this.hooks = [...this.hooks, ...added];
    return this;
  }

  /** Remove a hook by identity. Returns true if it was installed. */
  removeHook(hook: unknown): boolean {
    const before = this.hooks.length;
    this.hooks = this.hooks.filter((installed) => installed !== hook);
    return this.hooks.length !== before;
  }

  /**
   * Run `fn` with hooks installed, then remove them -- works for sync and async `fn`:
   *
   *     await router.withHooks([tracer], () => router.predict(state, questions));
   */
  withHooks<T>(hookArgs: HookArg[], fn: () => T): T {
    const added = normaliseHooks(hookArgs);
    this.hooks = [...this.hooks, ...added];
    const release = (): void => {
      this.hooks = this.hooks.filter((installed) => !added.includes(installed));
    };
    let out: T;
    try {
      out = fn();
    } catch (err) {
      release();
      throw err;
    }
    if (out instanceof Promise) {
      return out.finally(release) as T;
    }
    release();
    return out;
  }
}

/** Sum the per-state usage blocks so a hook sees one total for the call. */
export function aggregateUsage(results: Record<string, unknown>[]): Record<string, number> {
  const total = (key: string): number =>
    results.reduce((acc, r) => {
      const usage = r?.["usage"] as Record<string, unknown> | null | undefined;
      const v = usage?.[key];
      return acc + (typeof v === "number" && Number.isFinite(v) ? Math.trunc(v) : 0);
    }, 0);
  return { input_tokens: total("input_tokens"), output_tokens: total("output_tokens") };
}

function reportHookFailure(hook: Hook, event: HookEvent, err: unknown): void {
  const name = (hook as object)?.constructor?.name ?? "hook";
  console.warn(`laya: hook ${name}.${event} failed: ${String(err)}`);
}

/**
 * Call `event` on every hook that implements it.
 *
 * `raiseErrors=false` warns and continues, for hooks (telemetry) that must not fail a
 * request. Python also takes a `lock` to serialise dispatch for hooks that are not safe to
 * run concurrently; JS hooks run on one thread, so there is nothing to lock.
 *
 * It does not wait for a hook that returns a promise, so it is for callers that cannot wait,
 * such as the synchronous `Router.route`; a rejection can then only be reported as a warning.
 * `dispatchAsync` waits and applies `raiseErrors` to it.
 */

function isThenable(value: unknown): value is PromiseLike<unknown> {
  return typeof (value as PromiseLike<unknown> | null | undefined)?.then === "function";
}

/**
 * Python's `float()` read of a string, or null for a string `float()` rejects.
 *
 * `Number()` is looser in both directions: it reads `"inf"` as NaN where `float()` reads infinity,
 * and it reads `"0x10"` as 16 where `float()` raises. Decimal, exponent and the `inf`/`nan`
 * spellings are what a JSON body can hold; a `_`-grouped digit like `"1_000"`, which CPython does
 * read, is refused here as unconvertible rather than passed through `Number()`, which gives NaN.
 */
function floatFromText(text: string): number | null {
  const body = text.trim().toLowerCase();
  if (body === "nan" || body === "+nan" || body === "-nan") return NaN;
  if (/^[+-]?(inf|infinity)$/.test(body)) return body.startsWith("-") ? -Infinity : Infinity;
  if (!/^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/.test(body)) return null;
  return Number(body);
}

/**
 * The `hooksTimeout` of one call, in seconds: a positive finite number, or null for "no limit".
 * Port of `laya/hooks.py::validate_timeout` (hooks.py:251-285), whose refusals
 * `laya-java/fixtures/hooks.json` records under `timeouts`.
 *
 * The port had no gate at all -- `typeof value === "number" ? value * 1000 : null` -- so it
 * installed deadlines the reference refuses to install, and skipped the deadline for everything
 * not already a number. Measured on the built port with a hook whose `onPredictStart` never
 * settles, against the same table CPython's `validate_timeout` answers:
 *
 *     hooksTimeout   the reference                          laya-ts (before)
 *     0              ValueError, `got 0`                    TimeoutError at 1 ms; the hook never ran
 *     -1             ValueError, `got -1`                   TimeoutError at 2 ms
 *     NaN            ValueError, `got nan`                  TimeoutError at 2 ms
 *     Infinity       ValueError, `got inf`                  TimeoutError at 1 ms
 *     true           ValueError, `not a bool; got True`     no deadline, the hook runs unbounded
 *     [1]            ValueError, `got [1]`                  no deadline
 *     {"a": 1}       ValueError, `got {'a': 1}`             no deadline
 *     "0.5"          a 0.5 s deadline                       no deadline
 *
 * `setTimeout(0)` fires on the next tick and `setTimeout(-1000)` fires after 1 ms (Node warns
 * `TimeoutNegativeWarning` and clamps), so a zero or negative timeout is not a missing deadline
 * but an instant one: every hook call fails with a `TimeoutError` naming a deadline of `0s` that
 * no caller asked for, and with `hooksRaise: false` the prediction still completes and reports the
 * fake timeout as a warning. That is exactly the race Python's docstring refuses to leave to
 * `thread.join`. The non-number rows are the quieter half: a `hooksTimeout` read from a config
 * body as a string or a nested object silently became "no limit".
 */
export function validateHooksTimeout(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") {
    // `float(True)` is 1.0, so a bool would otherwise install a one-second deadline on every
    // hook call. Refused as a wrong type, not as a non-positive number.
    throw new Error(
      `hooks_timeout must be a positive finite number or None, not a bool; got ${pyRepr(value)}`,
    );
  }
  let seconds: number | null;
  if (typeof value === "number") seconds = value;
  else if (typeof value === "bigint") seconds = Number(value);
  else if (typeof value === "string") {
    const read = floatFromText(value);
    if (read === null) {
      // The message `float()` itself raises; the reference lets it escape from the coercion.
      throw new Error(`could not convert string to float: ${pyRepr(value)}`);
    }
    seconds = read;
  } else seconds = null;
  if (seconds === null) {
    // A list, an object, a function: `float()` raises TypeError, which the reference catches and
    // re-raises as the same ValueError every other bad timeout gets.
    throw new Error(`hooks_timeout must be a positive finite number or None; got ${pyRepr(value)}`);
  }
  if (!Number.isFinite(seconds) || seconds <= 0) {
    throw new Error(`hooks_timeout must be a positive finite number or None; got ${pyRepr(value)}`);
  }
  return seconds;
}

function trimTrailingZeros(decimal: string): string {
  const dot = decimal.indexOf(".");
  if (dot < 0) return decimal;
  let end = decimal.length;
  while (end > 0 && decimal.charAt(end - 1) === "0") end--;
  if (end > 0 && decimal.charAt(end - 1) === ".") end--;
  return decimal.slice(0, end);
}

/** Seconds the way CPython prints `"%g"`: `exceeded 0.05s` and `exceeded 1s`, not `1.0s`. */
function formatTimeoutSeconds(timeoutMs: number): string {
  const value = timeoutMs / 1000;
  if (!Number.isFinite(value)) return String(value);
  const scientific = value.toExponential(5);
  const marker = scientific.indexOf("e");
  if (marker < 0) return scientific;
  const exponent = Number(scientific.slice(marker + 1));
  if (exponent < -4 || exponent >= 6) {
    const sign = exponent < 0 ? "-" : "+";
    return (
      trimTrailingZeros(scientific.slice(0, marker)) +
      `e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`
    );
  }
  return trimTrailingZeros(value.toFixed(5 - exponent));
}

/** JS has no `TimeoutError` class; Python raises that builtin, so the name is set to match. */
function hookTimeoutError(operationName: string, timeoutMs: number): Error {
  const err = new Error(`laya: hook ${operationName} exceeded ${formatTimeoutSeconds(timeoutMs)}s`);
  err.name = "TimeoutError";
  return err;
}

/**
 * Await one hook result. The return value is discarded: nothing reads it.
 *
 * A non-thenable has already finished, so a configured timeout starts no timer. A thenable is
 * raced against `timeoutMs`, and the timer is cleared whether the hook settles or the deadline
 * wins. The deadline matches Python: `TimeoutError`, `laya: hook <Class.event> exceeded <N>s`.
 */
async function runWithTimeout(
  result: unknown,
  timeoutMs: number | null,
  operationName: string,
): Promise<void> {
  if (timeoutMs === null || timeoutMs === undefined) {
    await result;
    return;
  }
  if (!isThenable(result)) return;

  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      result,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          reject(hookTimeoutError(operationName, timeoutMs));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

export function dispatch(
  hooks: Hook[],
  event: HookEvent,
  ctx: PredictContext,
  opts: { raiseErrors?: boolean; timeoutMs?: number | null; hook?: Hook } = {},
): void {
  const raiseErrors = opts.raiseErrors ?? true;
  for (const hook of hooks) {
    const method = hook?.[event];
    if (typeof method !== "function") continue;
    try {
      const result: unknown = method.call(hook, ctx);
      if (isThenable(result)) {
        Promise.resolve(result).catch((err) => reportHookFailure(hook, event, err));
      }
    } catch (err) {
      if (raiseErrors) throw err;
      reportHookFailure(hook, event, err);
    }
  }
}

/** Like `dispatch`, but waits for each hook that returns a promise before the next runs, and a
 *  rejection follows `raiseErrors` the way a thrown error does. */
export async function dispatchAsync(
  hooks: Hook[],
  event: HookEvent,
  ctx: PredictContext,
  opts: { raiseErrors?: boolean; timeoutMs?: number | null; hook?: Hook } = {},
): Promise<void> {
  const timeoutMs = opts.timeoutMs ?? null;
  const raiseErrors = opts.raiseErrors ?? true;
  for (const hook of hooks) {
    const method = hook?.[event];
    if (typeof method !== "function") continue;
    const operationName = `${(hook as object)?.constructor?.name ?? "hook"}.${event}`;
    try {
      await runWithTimeout(method.call(hook, ctx), timeoutMs, operationName);
    } catch (err) {
      if (raiseErrors) throw err;
      reportHookFailure(hook, event, err);
    }
  }
}
