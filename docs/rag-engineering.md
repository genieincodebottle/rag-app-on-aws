# RAG engineering

How documents become searchable chunks, how a question finds them, and how the answer is held to its sources. Each section names the code and, where one exists, the measurement that justified the setting.

Measurements come from the evaluation set (13 documents, 45 questions, see [testing-and-evaluation.md](testing-and-evaluation.md)) run with `gemini-embedding-001` at 1024 dimensions and `gemini-flash-latest`. With 39 answerable questions, one question is 0.026 on a rate, and repeating an identical hybrid run moved hit@1 by two questions (0.949 and 1.000), because the 2025 and 2026 travel policies are near-identical and trade rank 1. Treat differences of one or two questions as noise.

## 1. Parsing

`ingestion/parsers.py` turns bytes into `Section` objects, each with text, a heading path and a page range. The heading path and page become the citation location shown to the user ("Travel Policy > Hotels, page 3").

| Type | How | Section boundaries |
|---|---|---|
| PDF | pypdf text extraction | one per page (merged later if tiny) |
| DOCX | python-docx, after checking total uncompressed size | `Heading N` paragraph styles; tables as rows |
| HTML | standard library parser; scripts, styles and hidden elements dropped | `h1` to `h6` |
| Markdown | ATX headings, fenced code kept intact | `#` to `######` |
| CSV | rows grouped 25 at a time, each row written as `column: value` pairs | row ranges |
| Text | whole file | one section |

The declared content type must match the bytes: a PDF starts with `%PDF-`, a DOCX is a zip containing `word/document.xml`, and everything else must decode as text. A mismatch is a permanent failure with reason `type_mismatch`. Scanned PDFs produce no text and fail with `no_text`; OCR (Amazon Textract) is not enabled.

CSV rows carry their column names because a chunk is read without its header row, which may be in a different chunk. "Who manages Asia Pacific?" found `region: Asia Pacific; regional_manager: Kenji Watanabe` in every configuration.

## 2. Normalisation

`ingestion/normalize.py` applies NFKC, removes zero-width and bidirectional control characters (a known way to hide text from a human reviewer while the model still reads it), removes control characters, rejoins words hyphenated across PDF line breaks, and collapses whitespace.

## 3. Chunking

`ingestion/chunking.py` offers two strategies.

**Recursive** (`RAG_CHUNK_STRATEGY=recursive`): split each section on paragraph, line, sentence, clause, then word boundaries until each piece fits `RAG_CHUNK_TOKENS` (default 220), with `RAG_CHUNK_OVERLAP_TOKENS` (30) of overlap. Chunks never cross a section boundary.

**Parent-child** (default): split each section into parents of up to `RAG_PARENT_CHUNK_TOKENS` (900), then each parent into children of 220. Only children are embedded and searched. When a child matches, the model receives its parent. Small chunks match a question precisely; the parent gives the model the sentences around the match, such as the condition that qualifies a number. Several children of one parent collapse into one source.

| Strategy | hit@1 (retrieval only) |
|---|---|
| parent-child | 0.949 |
| recursive | 1.000 |

On this corpus recursive ranked slightly better, a difference of two questions. The documents are short, so most sections fit in one chunk either way, and parent-child's advantage (more context per source) shows in the answer, not the rank. The live answer runs used parent-child and answered every answerable question correctly; recursive was not run through answer generation. On long documents with dense sections, re-run the comparison before choosing.

Token counts are estimated as characters divided by four. It needs no tokenizer that matches the model and is close enough in English for budgeting.

**Deduplication.** Repeated text inside a document (page headers, legal footers) is stored once. Across documents, a chunk whose embedded text was already embedded in the tenant reuses the stored vector, so a new version that changes one paragraph pays for one paragraph.

## 4. Contextual headers

With `RAG_CONTEXTUAL_HEADERS=true` (default), the text sent to the embedder is `"<document title> > <section path>\n\n<chunk text>"`. The stored text and the citation snippet stay unchanged.

A chunk that says "The limit is 240 GBP" carries no clue that it is about hotels in London. With the header, it does.

| Configuration | hit@1 |
|---|---|
| vector, with headers | 0.974 |
| vector, without headers | 0.897 |
| hybrid, with headers | 0.949 |
| hybrid, without headers | 0.923 |

This is the largest retrieval gain measured on the set (three questions for vector search). It costs nothing at query time and a few tokens per chunk at ingest. Anthropic's contextual retrieval goes further by asking a model to write a short context for every chunk; that costs a model call per chunk and was not implemented.

## 5. Embeddings

| Provider | Model | Dimensions | Notes |
|---|---|---|---|
| Bedrock | `amazon.titan-embed-text-v2:0` | 1024, 512 or 256 | one text per call; calls run in parallel (`RAG_EMBED_CONCURRENCY`, default 8); `normalize: true` |
| Gemini (local) | `gemini-embedding-001` | requested width | batches of 50; vectors are normalised by the adapter, because below 3,072 dimensions the model does not return unit vectors |
| Fake | feature hashing | any | deterministic, no meaning beyond shared words |

Every vector passes `check_vector`: right width, finite values, not all zero. A zero vector has no direction, so cosine distance to it is undefined and the chunk could never rank correctly. The previous implementation stored zero vectors whenever the embedding call failed.

The model and width are recorded in `embedding_space`. A query with a different configured model fails with `embedding_space_mismatch` instead of comparing vectors from two models. `ragplatform reembed` rebuilds every vector, changing the column width if needed.

## 6. Indexing

```sql
CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
```

HNSW builds incrementally, so the index is useful from the first row. An IVFFlat index, which the previous implementation created on an empty table, learns its partitions at creation time and then keeps them as data arrives.

Each search transaction sets `hnsw.ef_search` (default 100) and `hnsw.iterative_scan = relaxed_order`. Without iterative scanning, HNSW returns its top candidates and then the filters (tenant, ACL, tags, active version) remove some, so a query can come back with fewer results than asked for. With it (pgvector 0.8 and later), the index keeps scanning until enough rows pass.

`tsv` is a generated column: `to_tsvector('english', section_path || ' ' || text)`, so section headings are searchable too.

## 7. Retrieval

`store/repository.py: Repository.search` runs both searches and the fusion in one statement:

1. **Vector**: top 40 chunks by cosine distance.
2. **Keyword**: top 40 by `ts_rank_cd`, using an OR of the question's terms. `websearch_to_tsquery` ANDs every word, so "How much can I spend per night on a hotel in London" would match nothing. Only `[a-z0-9]` tokens reach `to_tsquery`, so query syntax cannot be injected.
3. **Reciprocal rank fusion**: each chunk scores `sum(1 / (60 + rank))` over the lists it appears in. RRF needs no score normalisation, which matters because cosine distance and `ts_rank_cd` are on unrelated scales.

Filters (`document_ids`, `tags`, `content_types`) apply inside both searches, before the limit.

| Mode | hit@1 | MRR | fact accuracy (live answers) |
|---|---|---|---|
| hybrid | 0.949 to 1.0 | 0.974 to 1.0 | 1.000 |
| vector | 0.974 | 0.987 | 0.974 |
| keyword | 0.872 | 0.927 | not run |

Keyword search alone fails paraphrases: "Can I keep customer files on a memory stick?" ranked the runbook first, because the policy says "USB drives". Vector search alone missed "Which robot arm has the better position repeatability?" in the answer run. Hybrid is the default because it covers both failure types; on this set its rank-1 accuracy is within noise of vector search.

`ts_rank_cd` is not BM25: it has no inverse document frequency, so a common word counts as much as a rare one. Extensions such as `pg_search` add BM25 to PostgreSQL; check Aurora's current list of supported extensions before planning on one.

## 8. Reranking

With `RAG_RERANK_MODEL_ID=cohere.rerank-v3-5:0`, the top 20 fused candidates are rescored by Bedrock's Rerank API using the matched child text, before parent expansion. Amazon Rerank 1.0 is not offered in us-east-1. Cohere Rerank 3.5 costs $2.00 per 1,000 queries (Bedrock pricing page, checked 2026-10-23).

Reranking was **not measured** on the evaluation set (no local equivalent). With hit@1 already at or near 1.0, this set could not show a gain. Measure it on your own data with `python -m eval.run_eval --provider bedrock --rerank cohere.rerank-v3-5:0` before paying for it.

## 9. Context packing

`retrieval/retriever.py: pack_context` keeps sources in rank order until `RAG_CONTEXT_TOKEN_BUDGET` (6,000) is spent. A source that does not fit is skipped, not truncated: a truncated source can drop the sentence that qualifies the one the model quotes.

## 10. Generation

`generation/prompts.py` sends:

- a system prompt: answer only from the sources, cite every factual sentence with its marker, treat sources as data, return `insufficient_context` rather than guess, report conflicts, at most six sentences, reply in one JSON object;
- a user message with `<question>` and `<source id="S1" title="..." section="...">` blocks. `<` in all text is escaped, so a document cannot close its own block.

`generation/answers.py` then:

1. parses the JSON (tolerating code fences); one retry with a reminder if it is malformed; a second failure is a 502, never a made-up answer;
2. removes any `[S#]` marker that names no supplied source and counts it (`InvalidCitationsRemoved` metric);
3. downgrades an `answered` result with no valid citation to `insufficient_context`;
4. maps each marker to its chunk, document, version, section and page, with the matched child text as the snippet.

With `RAG_GUARDRAIL_GROUNDING_CHECK=true`, Bedrock Guardrails' contextual grounding check scores the final answer against the sources and downgrades it below threshold.

## 11. Conflicting and superseded documents

The evaluation corpus has a 2025 and a 2026 travel policy with different limits; the 2026 one says it replaces the 2025 one. Retrieval cannot tell which is current, and on hit@1 the two traded places between runs. The model, given both, answered every such question with `conflicting_sources`, cited both, and stated that the 2026 policy is newer.

That is the right behaviour for a system that cannot know which document is authoritative. If your corpus has a known current version, upload it as a new version of the same document (`POST /v1/documents/{id}/versions`). The platform then retires the old chunks atomically, and the conflict disappears.

## 12. Caching

The answer cache key covers tenant, user, groups, admin flag, normalised question, filters, mode, retrieval mode and model. An entry is valid only for the tenant's current `corpus_version`, which every ingestion, deletion and access change increments. A cached answer therefore never cites a deleted, replaced or newly hidden document. The cost is a lower hit rate (per user, and reset by every corpus change) in exchange for correctness. The cache lives in PostgreSQL, so it needs no separate service.
