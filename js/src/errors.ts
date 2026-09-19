/**
 * Error classes for the Laya SDK.
 * @module errors
 */

/** Base class for all Laya-related errors. */
export class LayaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** Thrown when the Laya API returns a 4xx or 5xx status code. */
export class LayaAPIError extends LayaError {
  public status: number;
  public code: string;
  public param?: string;

  constructor(status: number, code: string, message: string, param?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.param = param;
  }
}

/** Thrown when a request to the Laya API exceeds the configured timeout. */
export class LayaTimeoutError extends LayaError {
  constructor(message: string = 'Request timed out') {
    super(message);
  }
}

/** Thrown when a request to the Laya API fails due to network issues. */
export class LayaNetworkError extends LayaError {
  constructor(message: string) {
    super(message);
  }
}
