# MiniBug curriculum data

## Global context

`minibug_tasks.json` is a repository-authored curriculum declared clean-room for verifiable Python function repair. The repository does not provide an independent originality audit. The curriculum was fixed in the [source commit that produced the reference run](https://github.com/BurnyCoder/llm-rl-software-engineering/commit/7027bc55baecc00fad51cbe2b8f030dca2c91c1e) and contains exactly 60 deterministic tasks: 36 training tasks, 12 validation tasks, and 12 final-test tasks. The records contain buggy programs and expected behavior, but no gold repairs. This keeps a canonical repaired implementation out of the prompt while allowing a test-based reward to be computed later in an isolated executor.

Each task is one short, import-free function and has two public examples. In the frozen file, 59 tasks have four hidden cases and one has five; the validator schema permits four through eight. Arguments, keyword arguments, and expected values use only JSON values so the later sandbox boundary can use a narrow serialization format. The format follows Python 3.12's documented [`json` conversion rules](https://docs.python.org/3.12/library/json.html#encoders-and-decoders).

## Record contract

Every task has `id`, `split`, `family`, `difficulty`, `specification`, `function_name`, `buggy_code`, `public_tests`, and `hidden_tests`. Every test case has an `args` list, a `kwargs` object, and an `expected` JSON value. IDs are unique across the complete curriculum.

`split_manifest.json` makes accidental edits detectable. A task's `canonical_sha256` hashes its UTF-8 serialization produced with `ensure_ascii=False`, `sort_keys=True`, and compact separators; Python documents sorted keys as useful for stable regression comparisons. The `ast_fingerprint` hashes [`ast.dump`](https://docs.python.org/3.12/library/ast.html#ast.dump) of [`ast.parse`](https://docs.python.org/3.12/library/ast.html#ast.parse) with source-location attributes excluded. Before hashing, function-bound names are alpha-renamed and literals are replaced by type-specific sentinels. Control flow, operators, free function names, attributes, and literal types remain, so renamed or reparameterized templates collide without treating `sum(x)` and `bool(x)` as the same operation. Digests use the standard library's documented [`hashlib.sha256`](https://docs.python.org/3.12/library/hashlib.html#hashlib.sha256).

## Leakage controls

- The final manifest and 36/12/12 split were frozen before the authoritative run; earlier diagnostic history does not prove assignment before every project experiment.
- No normalized structural fingerprint appears in more than one split.
- No public case exactly duplicates a hidden case within any task; the curriculum test
  compares canonical JSON records and rejects an overlap.
- Pre-training structural checks and original-bug sandbox canaries cover every internal split. Validation model outcomes select the checkpoint and budget; final-test model scoring begins only after that lock and cannot change it, as recorded in the checked [run evidence](../reports/evidence/run-summary.json).
- Public cases may appear in prompts. Hidden cases are reward/evaluation inputs and must not be placed in prompts or generation logs.
- Because expected outputs necessarily reveal behavioral examples, neither public nor hidden cases should be described as gold implementations.

The curriculum is intentionally small and synthetic. It measures local function-repair behavior, not repository-scale software engineering, and its results should not be generalized beyond that scope.
