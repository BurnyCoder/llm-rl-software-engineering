# MiniBug curriculum data

## Global context

`minibug_tasks.json` is an original, clean-room curriculum for verifiable Python function repair. It contains exactly 60 deterministic tasks: 36 training tasks, 12 validation tasks, and 12 final-test tasks. The records contain buggy programs and expected behavior, but no gold repairs. This keeps the policy target out of the prompt while allowing a test-based reward to be computed later in an isolated executor.

Each task is one short, import-free function and has two public examples plus four to eight hidden cases. Arguments, keyword arguments, and expected values use only JSON values so the later sandbox boundary can use a narrow serialization format. The format follows Python's documented [`json` conversion rules](https://docs.python.org/3/library/json.html#encoders-and-decoders).

## Record contract

Every task has `id`, `split`, `family`, `difficulty`, `specification`, `function_name`, `buggy_code`, `public_tests`, and `hidden_tests`. Every test case has an `args` list, a `kwargs` object, and an `expected` JSON value. IDs are unique across the complete curriculum.

`split_manifest.json` makes accidental edits detectable. A task's `canonical_sha256` hashes its UTF-8 serialization produced with `ensure_ascii=False`, `sort_keys=True`, and compact separators; Python documents sorted keys as useful for stable regression comparisons. The `ast_fingerprint` hashes [`ast.dump`](https://docs.python.org/3/library/ast.html#ast.dump) of [`ast.parse`](https://docs.python.org/3/library/ast.html#ast.parse) with source-location attributes excluded. Before hashing, function-bound names are alpha-renamed and literals are replaced by type-specific sentinels. Control flow, operators, free function names, attributes, and literal types remain, so renamed or reparameterized templates collide without treating `sum(x)` and `bool(x)` as the same operation. Digests use the standard library's documented [`hashlib.sha256`](https://docs.python.org/3/library/hashlib.html#hashlib.sha256).

## Leakage controls

- Splits are assigned before experiments and recorded in the manifest.
- No normalized structural fingerprint appears in more than one split.
- Validation is for checkpoint selection; final-test records are reserved for the single locked evaluation.
- Public cases may appear in prompts. Hidden cases are reward/evaluation inputs and must not be placed in prompts or generation logs.
- Because expected outputs necessarily reveal behavioral examples, neither public nor hidden cases should be described as gold implementations.

The curriculum is intentionally small and synthetic. It measures local function-repair behavior, not repository-scale software engineering, and its results should not be generalized beyond that scope.
