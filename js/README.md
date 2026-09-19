# laya-sdk

[![npm](https://img.shields.io/npm/v/laya-sdk)](https://npmjs.com/package/laya-sdk)
[![bundle size](https://img.shields.io/bundlephobia/minzip/laya-sdk)](https://bundlephobia.com/package/laya-sdk)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue)](https://typescriptlang.org)
[![Drop-in for @typesafe-ai/sdk](https://img.shields.io/badge/drop--in%20for-%40typesafe--ai%2Fsdk-orange)](https://github.com/NandhaKishorM/laya)

JavaScript/TypeScript SDK for Laya — open-source drop-in replacement for TypeSafe Jev.

## Installation

```bash
npm install laya-sdk
```

## Migration Guide

Migrating from `@typesafe-ai/sdk` requires changing only the import statement and passing your self-hosted Laya server URL.

**Before:**
```typescript
import { TypeSafeClient, choice, score, noul } from "@typesafe-ai/sdk";
const client = new TypeSafeClient();
```

**After:**
```typescript
import { TypeSafeClient, choice, score, noul } from "laya-sdk";
const client = new TypeSafeClient({ baseURL: "http://localhost:8000" });
```

All method names, arguments, and response types are completely identical.

## Quickstart

```typescript
import { LayaClient, choice, score, noul } from "laya-sdk";

const client = new LayaClient({ baseURL: 'http://localhost:8000' });

const { answers } = await client.systemOne({
  state: { body: 'I was charged twice!' },
  questions: {
    department: choice('Which team?', { billing: 'invoices', technical: 'bugs' }),
    isUrgent: noul('Is this urgent?'),
    frustration: score('How frustrated?', ['calm', 'concerned', 'very angry']),
  },
});

console.log(answers.department.choice);    // 'billing'
console.log(answers.isUrgent.noul);        // 0.92
console.log(answers.frustration.score);    // 1.84
```

## Batch API Example

```typescript
const { results } = await client.decideBatch({
  states: [{ text: "Hello" }, { text: "Refund me!" }],
  questions: {
    intent: choice("What is the user's intent?", { greeting: "greeting", refund: "refund" })
  }
});

console.log(results[0].answers.intent.choice); // "greeting"
console.log(results[1].answers.intent.choice); // "refund"
```

## Client Options

| Option       | Type     | Default                 | Description                                    |
|--------------|----------|-------------------------|------------------------------------------------|
| `baseURL`    | `string` | `http://localhost:8000` | The base URL of your Laya server               |
| `apiKey`     | `string` | `undefined`             | Optional API key for auth (Bearer token)       |
| `timeout`    | `number` | `30000`                 | Request timeout in milliseconds                |
| `retries`    | `number` | `2`                     | Number of automatic retries on 5xx/network err |
| `retryDelay` | `number` | `500`                   | Base delay for exponential backoff (ms)        |

Environment Variables: `LAYA_BASE_URL` will be used if `baseURL` is not provided.

## Error Handling

The SDK provides structured errors for robust handling:

```typescript
import { LayaAPIError, LayaTimeoutError, LayaNetworkError } from "laya-sdk";

try {
  await client.systemOne({ /* ... */ });
} catch (error) {
  if (error instanceof LayaAPIError) {
    console.error(`API Error ${error.status}: ${error.code} - ${error.message}`);
  } else if (error instanceof LayaTimeoutError) {
    console.error("Request timed out");
  } else if (error instanceof LayaNetworkError) {
    console.error("Network issue");
  }
}
```
