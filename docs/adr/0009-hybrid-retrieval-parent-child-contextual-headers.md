# 0009. Hybrid retrieval with parent-child chunks and contextual headers by default

- Status: accepted, 2026-10-24

## Context

Defaults for retrieval should be justified by measurement on a labelled set, not by habit.

## Decision

Default to hybrid retrieval (vector and keyword fused with reciprocal rank fusion), parent-child chunking (220-token children searched, 900-token parents sent to the model) and contextual headers (title and section path prefixed to embedded text). Reranking is available and off by default.

## Evidence

From [testing-and-evaluation.md](../testing-and-evaluation.md), Gemini embeddings, 39 answerable questions:

- Contextual headers: vector hit@1 0.897 without, 0.974 with. The clearest gain measured.
- Keyword only: hit@1 0.872; misses paraphrases. Vector only: hit@1 0.974, but missed one cross-document comparison in the answer run (fact accuracy 0.974 against 1.000 for hybrid).
- Hybrid and vector are within noise of each other on rank; hybrid is kept because it covers both failure types (paraphrase for keyword, exact terms for vector).
- Recursive chunking ranked as well as parent-child on these short documents; parent-child is kept for the context it gives the model on longer documents. This should be re-measured on long documents.
- Reranking: not measured.

## Consequences

- Each setting is one environment variable, and the evaluation runner compares them in minutes.
- The defaults should be re-measured on each real corpus; the evaluation set's documents are short.
