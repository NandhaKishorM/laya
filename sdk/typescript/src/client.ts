import {
  LayaAbortError, LayaAPIError, LayaConnectionError, LayaError, LayaResponseError,
  LayaTimeoutError, LayaValidationError,
} from './errors.js';
import type { Health, Prediction, PredictOptions, Questions, RequestOptions, RouteDecision, State } from './types.js';
import { isRecord, validateJson, validateQuestions, validateTimeout } from './validation.js';
import { validateHealth, validatePrediction, validateRoute } from './response.js';

export interface LayaOptions {
  /** Server root URL, optionally including a reverse-proxy path prefix. */
  baseURL?: string;
  apiKey?: string;
  /** Default 120000 ms; cold checkpoint downloads may require more. Zero disables. */
  timeoutMs?: number;
  headers?: HeadersInit;
  /** Supply a Fetch-compatible implementation for tests or a custom transport. */
  fetch?: typeof globalThis.fetch;
}

export class Laya {
  private readonly baseURL: string;
  private readonly timeoutMs: number;
  private readonly headers: Headers;
  private readonly fetcher: typeof globalThis.fetch;

  constructor(options: LayaOptions = {}) {
    let url: URL;
    try { url = new URL(options.baseURL ?? 'http://127.0.0.1:8000'); }
    catch { throw new LayaValidationError('baseURL must be an absolute HTTP(S) URL'); }
    if (!['http:', 'https:'].includes(url.protocol) || url.search || url.hash || url.username || url.password) {
      throw new LayaValidationError('baseURL must be HTTP(S), without credentials, query, or fragment');
    }
    this.baseURL = url.href.replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? 120_000;
    validateTimeout(this.timeoutMs);
    this.headers = new Headers(options.headers);
    this.headers.set('Accept', 'application/json');
    if (options.apiKey !== undefined) this.headers.set('Authorization', `Bearer ${options.apiKey}`);
    const fetcher = options.fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') throw new LayaValidationError('A Fetch implementation is required');
    this.fetcher = fetcher;
  }

  async predict<const Q extends Questions>(state: State, questions: Q, options: PredictOptions = {}): Promise<Prediction<Q>> {
    validateQuestions(questions, false);
    const result = await this.request<Prediction<Q>>('/v1/predict', this.body(state, questions, options), options);
    validatePrediction(result, questions);
    return result;
  }

  async route(state: State, questions: Questions = {}, options: PredictOptions = {}): Promise<RouteDecision> {
    validateQuestions(questions, true);
    const result = await this.request<RouteDecision>('/v1/route', this.body(state, questions, options), options);
    validateRoute(result);
    return result;
  }

  async health(options: RequestOptions = {}): Promise<Health> {
    const result = await this.request<Health>('/health', undefined, options);
    validateHealth(result);
    return result;
  }

  private body(state: State, questions: Questions, options: PredictOptions): string {
    if (state !== null && typeof state !== 'string' && !Array.isArray(state) && !isRecord(state)) {
      throw new LayaValidationError('state must be text, a JSON object, an array, or null');
    }
    const body: Record<string, unknown> = { state, questions };
    for (const key of ['model', 'task', 'lang'] as const) {
      if (options[key] !== undefined) {
        if (typeof options[key] !== 'string') throw new LayaValidationError(`${key} must be a string`);
        body[key] = options[key];
      }
    }
    validateJson(body);
    return JSON.stringify(body);
  }

  private async request<T>(path: string, body: string | undefined, options: RequestOptions): Promise<T> {
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    validateTimeout(timeoutMs);
    const controller = new AbortController();
    const abort = () => controller.abort(new LayaAbortError('Request aborted', { cause: options.signal?.reason }));
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) abort();
    const timer = timeoutMs > 0 ? setTimeout(() => {
      controller.abort(new LayaTimeoutError(`Request timed out after ${timeoutMs} ms`));
    }, timeoutMs) : undefined;

    try {
      controller.signal.throwIfAborted();
      const headers = new Headers(this.headers);
      if (body !== undefined) headers.set('Content-Type', 'application/json');
      const init: RequestInit = { method: body === undefined ? 'GET' : 'POST', headers, signal: controller.signal };
      if (body !== undefined) init.body = body;
      // Never retry automatically: repeating an expensive inference request is surprising.
      const response = await this.fetcher.call(globalThis, this.baseURL + path, init);
      const raw = await response.text();
      let payload: unknown;
      try { payload = JSON.parse(raw); }
      catch {
        if (response.ok) throw new LayaResponseError('Server returned invalid JSON');
      }
      if (!response.ok) {
        const error = isRecord(payload) && isRecord(payload.error) ? payload.error : undefined;
        throw new LayaAPIError(
          typeof error?.message === 'string' ? error.message : `Laya returned HTTP ${response.status}`,
          response.status,
          typeof error?.code === 'string' ? error.code : 'http_error',
          error?.details ?? payload ?? raw,
        );
      }
      if (!isRecord(payload)) throw new LayaResponseError('Server returned a non-object JSON response');
      return payload as T;
    } catch (error) {
      if (controller.signal.aborted) throw controller.signal.reason;
      if (error instanceof LayaError) throw error;
      throw new LayaConnectionError('Could not reach the Laya server', { cause: error });
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      options.signal?.removeEventListener('abort', abort);
    }
  }
}
