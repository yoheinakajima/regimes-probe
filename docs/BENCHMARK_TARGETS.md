# Benchmark Targets

`regimes-probe` evaluates **procedural epistemic policy** on browsing benchmarks. The
targets are chosen to minimize intrinsic-knowledge leakage (so improvements reflect
*search behavior*, not memorized facts) and to be cheap and unambiguous to grade.

## Primary — LiveBrowseComp

Recent-facts browsing benchmark. Because answers turn on **recent** events, the answerer
cannot rely on parametric/intrinsic knowledge, which is exactly the regime where learned
search policy should pay off (freshness-sensitive, staleness-prone signatures).

- Paper: <https://arxiv.org/abs/2605.28721>
- Dataset: <https://huggingface.co/datasets/Forival/LiveBrowseComp>
- Adapter: `LiveBrowseCompAdapter`
- Config: `benchmark_primary=LiveBrowseComp`

## Secondary — BrowseComp

Public, short-answer browsing benchmark. Short answers make grading easy and reduce grader
variance, giving a stable secondary signal.

- Paper: <https://arxiv.org/abs/2504.12516>
- Eval harness: <https://github.com/openai/simple-evals>
- Reference grader: <https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py>
- Adapter: `BrowseCompAdapter`
- Config: `benchmark_secondary=BrowseComp`

> **Obfuscation note.** BrowseComp ships question/answer fields **obfuscated**
> (XOR/base64 canary-style encoding) to deter scraping/training leakage. The adapter must
> decode rows at load time; see *Decryption / obfuscation* below.

## Optional future — BrowseComp-Plus

Fixed-corpus, controlled-retrieval variant. Useful later for ablations that need a closed,
controlled document set (deterministic retrieval, no live web variance).

- Paper: <https://arxiv.org/abs/2508.06600>
- Adapter: *(future)* fixed-corpus adapter; not required for Studies 0–5.

## Dataset adapters

All adapters present a common `benchmark_item` interface to the harness and emit
`item.queued` events. They differ only in source and decode logic.

| Adapter | Source | Notes |
| --- | --- | --- |
| `SyntheticBrowseAdapter` | local fixtures | deterministic, oracle-labeled, **no network/API keys**; powers Study 0 and all unit tests |
| `BrowseCompAdapter` | BrowseComp (simple-evals) | decodes XOR/base64 canary obfuscation; short-answer grading |
| `LiveBrowseCompAdapter` | HF `Forival/LiveBrowseComp` | recent-facts; primary target |

### Unit-test policy

Unit tests **must not require real benchmark data**. Tests run exclusively against
`SyntheticBrowseAdapter` with deterministic fake tools and `HashEmbedder`. `BrowseComp` and
`LiveBrowseComp` loaders are exercised only in opt-in integration tests that skip cleanly
when data/keys are absent.

## Checksum / version capture

Every loaded dataset records provenance so a `benchmark_run` is reproducible and so we can
detect silent upstream changes:

- `dataset_name`, `dataset_version` / HF revision (commit SHA or tag),
- per-file **SHA-256 checksum** of the raw downloaded artifact,
- row count and a stable per-row id,
- adapter version,
- load timestamp (recorded at the event boundary, not generated inside a behavior).

```python
# Provenance captured at load time (illustrative, lives in src/regimes_probe/datasets/)
DatasetProvenance(
    dataset_name="LiveBrowseComp",
    dataset_version="<hf_revision_sha>",
    sha256="<sha256_of_raw_artifact>",
    row_count=...,
    adapter="LiveBrowseCompAdapter",
    adapter_version="...",
)
```

The provenance is attached to the `benchmark_run` and surfaced in the `report`. A run whose
checksum/version differs from a prior run is flagged so results are not silently compared
across dataset versions.

## Decryption / obfuscation

BrowseComp obfuscates `question` and `answer` fields (XOR against a per-row **canary** key,
base64-wrapped), following the simple-evals scheme. The adapter exposes a graceful loader:

```python
def decrypt(answer: str, canary: str) -> str:
    """Decode a BrowseComp-obfuscated field.

    Mirrors the simple-evals XOR/base64 canary scheme:
      raw = base64decode(field); plaintext = xor(raw, repeat(canary))
    Returns the decoded plaintext. Raises DecodeError on malformed rows.
    """
```

**Fail-gracefully requirement.** If a row cannot be decoded (truncated base64, wrong
canary, length mismatch), the loader must:

- **skip the row**, never crash the run,
- record a structured warning (`row_id`, reason) in the `benchmark_run` / `report`,
- count skipped rows so the effective evaluated set is transparent.

This keeps a partially-corrupt download from invalidating an entire `benchmark_run` while
making the omission auditable.
