export class LayaError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** Request rejected locally, before any network traffic. */
export class LayaValidationError extends LayaError {}

/** A non-2xx response. Includes JSON error details when available. */
export class LayaAPIError extends LayaError {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code: string,
    public readonly details: unknown,
  ) {
    super(message);
  }
}

export class LayaConnectionError extends LayaError {}
export class LayaTimeoutError extends LayaError {}
export class LayaAbortError extends LayaError {}
export class LayaResponseError extends LayaError {}
