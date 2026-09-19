/**
 * LayaClient implementation.
 * @module client
 */

import {
  LayaClientOptions,
  DecideRequest,
  DecideResponse,
  BatchDecideRequest,
  BatchDecideResponse,
  HealthResponse,
  ModelsResponse,
} from './types';
import { fetchWithRetry } from './utils';

/**
 * Client for interacting with the Laya API.
 */
export class LayaClient {
  private baseURL: string;
  private apiKey?: string;
  private timeout: number;
  private retries: number;
  private retryDelay: number;
  private fetchFn?: typeof fetch;

  /**
   * Initializes a new LayaClient.
   * @param options - Configuration options.
   */
  constructor(options?: LayaClientOptions) {
    this.baseURL =
      options?.baseURL ||
      (typeof process !== 'undefined' && process.env?.LAYA_BASE_URL) ||
      'http://localhost:8000';
      
    // Trim trailing slash
    if (this.baseURL.endsWith('/')) {
      this.baseURL = this.baseURL.slice(0, -1);
    }

    this.apiKey = options?.apiKey;
    this.timeout = options?.timeout ?? 30000;
    this.retries = options?.retries ?? 2;
    this.retryDelay = options?.retryDelay ?? 500;
    this.fetchFn = options?.fetch;
  }

  /**
   * Evaluates questions against a state.
   * Identical to the @typesafe-ai/sdk systemOne method.
   * @param request - The decision request.
   * @returns The decision response.
   */
  public async systemOne(request: DecideRequest): Promise<DecideResponse> {
    return this.decide(request);
  }

  /**
   * Evaluates questions against a state.
   * Alias for systemOne.
   * @param request - The decision request.
   * @returns The decision response.
   */
  public async decide(request: DecideRequest): Promise<DecideResponse> {
    return this.request<DecideResponse>('/v1/decide', 'POST', request);
  }

  /**
   * Evaluates questions against multiple states in a batch.
   * @param request - The batch decision request.
   * @returns The batch decision response.
   */
  public async decideBatch(request: BatchDecideRequest): Promise<BatchDecideResponse> {
    return this.request<BatchDecideResponse>('/v1/decide/batch', 'POST', request);
  }

  /**
   * Checks the health of the Laya server.
   * @returns The health status.
   */
  public async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health', 'GET');
  }

  /**
   * Retrieves the available models.
   * @returns The models response.
   */
  public async models(): Promise<ModelsResponse> {
    return this.request<ModelsResponse>('/v1/models', 'GET');
  }

  private async request<T>(path: string, method: string, body?: unknown): Promise<T> {
    const url = `${this.baseURL}${path}`;
    const headers: Record<string, string> = {};

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const options: RequestInit = {
      method,
      headers,
    };

    if (body) {
      options.body = JSON.stringify(body);
    }

    return fetchWithRetry<T>({
      url,
      options,
      fetchFn: this.fetchFn,
      timeout: this.timeout,
      retries: this.retries,
      retryDelay: this.retryDelay,
    });
  }
}
