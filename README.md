# vectorizer-napi

**NAPI-RS** addon for text embedding: call Rust (rust-bert, all-MiniLM-L6-v2) directly from Node.js with no HTTP service. Standalone — no parent crate required.

**Repository:** [https://github.com/MaloLebrin/vectorizer-napi](https://github.com/MaloLebrin/vectorizer-napi)

- **[Full NAPI-RS documentation](docs/NAPI-RS.md)** — architecture, API, examples, build, troubleshooting
- [NAPI-RS — Getting started](https://napi.rs/docs/introduction/getting-started)
- Computation (rust-bert) runs in the libuv thread pool via **AsyncTask** so the main thread is not blocked.

## Prerequisites

- **Rust** (rustup) and **Node.js** ≥ 18
- **libtorch** (for rust-bert / tch-rs): [tch-rs](https://github.com/LaurentMazare/tch-rs)

## Installation

```bash
cd vectorizer-napi
npm install
npm run build
```

The build produces `index.js`, `index.d.ts`, and `vectorizer.<platform>.node` (e.g. `vectorizer.darwin-x64.node`).

## Usage

```javascript
const { vectorize, modelName } = require('./index.js');

(async () => {
  console.log('Model:', modelName()); // "all-MiniLM-L6-v2"
  const embedding = await vectorize('Full-stack Node.js developer, 3 years experience.');
  console.log('Dimensions:', embedding.length); // 384
  console.log('First values:', embedding.subarray(0, 3));
})();
```

TypeScript:

```typescript
import { vectorize, modelName } from './index.js';

const embedding: Float64Array = await vectorize('Your text here');
```

## Integration in your app

In your Node/TypeScript app:

```typescript
import { vectorize } from 'vectorizer-napi';

export async function vectorizeText(text: string): Promise<number[]> {
  const arr = await vectorize(text);
  return Array.from(arr);
}
```

No HTTP service or `child_process` needed; everything runs in the same process.

## Scripts

- `npm run build` — release build with binaries for the platforms in `package.json` > `napi.targets`
- `npm run build:debug` — debug build (faster, for development)

## Targets

Default: `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`, `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-gnu`, `aarch64-unknown-linux-musl`. Adjust in `package.json` > `napi.targets` as needed.

## Repository

- **Git:** [https://github.com/MaloLebrin/vectorizer-napi](https://github.com/MaloLebrin/vectorizer-napi)

This repo is **standalone**: clone and build with no extra crates or submodules.

```bash
git clone https://github.com/MaloLebrin/vectorizer-napi.git
cd vectorizer-napi
npm install
npm run build
```

## Documentation

- **[docs/NAPI-RS.md](docs/NAPI-RS.md)** — detailed NAPI-RS doc: concepts (AsyncTask, TypedArray, thread-local model), API reference, JS/TS examples, build, troubleshooting, links to napi.rs.
