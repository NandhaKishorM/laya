/**
 * Utility functions for the Laya SDK.
 * @module utils
 */

import { LayaAPIError, LayaNetworkError, LayaTimeoutError } from './errors';

interface FetchWithRetryOptions {
  url: string;
  options: RequestInit;
  fetchFn?: typeof fetch;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
}

/**
 * Generates a random UUID v4 for the X-Request-Id header.
 */
function uuidv4(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0,
      v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Performs a fetch with timeout, automatic retries with exponential backoff on 5xx errors or network errors.
 * @param options - Configuration options for fetch.
 * @returns The JSON response data.
 */
export async function fetchWithRetry<T>(config: FetchWithRetryOptions): Promise<T> {
  const { url, options, fetchFn = globalThis.fetch, timeout = 30000, retries = 2, retryDelay = 500 } = config;

  let attempt = 0;
  
  // Clone headers and ensure X-Request-Id is set
  const headers = new Headers(options.headers);
  if (!headers.has('X-Request-Id')) {
    headers.set('X-Request-Id', uuidv4());
  }
  
  if (!headers.has('Content-Type') && options.body) {
    headers.set('Content-Type', 'application/json');
  }

  while (attempt <= retries) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetchFn(url, {
        ...options,
        headers,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (response.ok) {
        return (await response.json()) as T;
      }

      const status = response.status;
      let errorBody: any;
      try {
        errorBody = await response.json();
      } catch (e) {
        errorBody = { message: await response.text() };
      }

      if (status >= 500 && attempt < retries) {
        // Retry on 5xx
        attempt++;
        await new Promise((resolve) => setTimeout(resolve, retryDelay * Math.pow(2, attempt - 1)));
        continue;
      }

      const code = errorBody?.code || errorBody?.error?.code || 'unknown_error';
      const message = errorBody?.message || errorBody?.error?.message || `HTTP error ${status}`;
      const param = errorBody?.param || errorBody?.error?.param;

      throw new LayaAPIError(status, code, message, param);
    } catch (error: any) {
      clearTimeout(timeoutId);

      if (error instanceof LayaAPIError) {
        throw error;
      }

      if (error.name === 'AbortError') {
        throw new LayaTimeoutError(`Request timed out after ${timeout}ms`);
      }

      if (attempt < retries) {
        attempt++;
        await new Promise((resolve) => setTimeout(resolve, retryDelay * Math.pow(2, attempt - 1)));
        continue;
      }

      throw new LayaNetworkError(error.message || 'Network error');
    }
  }
  
  throw new LayaNetworkError('Max retries exceeded'); // Should be unreachable
}
