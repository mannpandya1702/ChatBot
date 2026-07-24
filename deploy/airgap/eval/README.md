# Golden eval set

A tiny release gate for the chat pipeline. It proves the two invariants that
matter most after you ingest or change the knowledge base:

- **Grounded questions get cited answers.** An in-KB question returns
  `refused: false` with at least one `[S#]` citation.
- **Everything else refuses.** Out-of-scope questions and prompt-injection
  attempts never produce a grounded, cited answer — no citations at all.

## Files

- `golden.example.jsonl` — a starter set. The `oos-*` and `inj-*` cases are
  content-independent and should pass on any deployment. The `kb-*` cases are
  **placeholders you must fill in** with questions your own documents answer.
- `run-eval.mjs` — a zero-dependency runner (Node 18+). Non-zero exit on any
  failure, so CI or a pre-go-live check can gate on it.

## Use it

1. Copy the example and fill in the `kb-*` questions with real ones your
   ingested documents answer (and replace `expect` if needed):
   ```bash
   cp deploy/airgap/eval/golden.example.jsonl deploy/airgap/eval/golden.jsonl
   # edit golden.jsonl — replace the "REPLACE ME" questions
   ```
2. Get an access token for a signed-in user (any active jawan/admin).
3. Run it:
   ```bash
   CHAT_URL=http://localhost:3000/api/chat ACCESS_TOKEN=<jwt> \
     node deploy/airgap/eval/run-eval.mjs deploy/airgap/eval/golden.jsonl
   ```

Keep each run under 20 questions — the pipeline rate-limits at 20 messages per
5 minutes per user. If a grounded case unexpectedly refuses, it is usually the
rerank gate, not a bug: confirm the document answers the question in words a
user would actually type (acronyms matter). See `../README.md` §11.

> Do not put restricted document text into `golden.jsonl` if that file will
> leave the air-gapped host. Keep the filled-in set on the deployment machine.
