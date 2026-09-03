# MiniBug-RL contributor instructions

Keep the smallest end-to-end repair-RL path working before adding optional features. Use `uv`, the local `.venv`, test-driven development, modular phase functions, timestamped untruncated prompt/completion logs, and primary-source comments. Never execute generated code on the host and never commit `.env`, model weights, caches, or raw run logs.

The public entry point must remain a thin coordinator over independently testable `preflight`, `prepare`, `baseline`, `smoke`, `train`, `evaluate`, `export`, and `publish` phases. Documentation, experiment reports, and measured claims must match checked artifacts and commands.
