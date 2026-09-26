# Ollama service (Railway)

A standalone Railway service running [Ollama](https://ollama.com). `besc-engagement-checker` calls this over Railway's private network to generate creative rewrite candidates. Every candidate still gets scored and gated through the app's own deterministic scorer, so this only ever *suggests*, never decides.

The model is pulled at container startup (`start.sh`), not baked into the image at build time. Baking it in via a backgrounded-server build trick turned out unreliable in practice (the image shipped with zero blobs despite the build step appearing to succeed). Attaching a Railway Volume keeps the pulled model persistent across restarts, so only the first boot is slow.

## Setup (Railway dashboard)

1. **New Service → Deploy from GitHub repo**, pick this same repo (`BESCLLC/x-algo-BESC-twitter-engagement-`).
2. **Name the service `ollama`.** This becomes its internal hostname (`ollama.railway.internal`). A different name works too, just adjust `OLLAMA_URL` in step 6 to match.
3. **Settings → Source → Root Directory**: set to `ollama-service`. Railway auto-detects the `Dockerfile` here and builds it directly, no `railway.json` needed.
4. **Attach a Volume** (Settings → Volumes → New Volume): mount path `/root/.ollama`, sized at least 6GB for the default `llama3.2:3b` model (~2GB) plus headroom. Without a volume, the model re-downloads on every restart, which still works but is slow every time.
5. **Sizing**: give this service at least 4GB RAM (Settings → Resources, or your plan's equivalent). `qwen2.5:14b-instruct` and `qwen2.5:7b-instruct` both appeared to hit a hard 100s timeout on first try — turned out to be cold-start (the first `/api/generate` call after boot pays for paging model weights into RAM, and that competes with the app's request timeout), not a CPU ceiling. Once a model is warm, requests are fast and consistent. Give a freshly-deployed or freshly-restarted service one throwaway request before judging it slow.
6. **Deploy**, then check the deploy logs for `Pulling model ...` output. The first boot with an empty volume takes a minute or two; watch for it to finish before testing. Once it's live, go to the **besc-engagement-checker** service's Variables and add:
   - `OLLAMA_URL=http://ollama.railway.internal:11434`
   - `OLLAMA_MODEL=qwen2.5:7b-instruct` (match whatever `ARG MODEL` you built with)

If the app can't reach it, double check **Private Networking** is enabled on both services (Railway project settings). It's on by default for services created in the same project, but worth a quick check.

## Verifying the model actually persisted

After a deploy, check the service logs for a line like `msg="total blobs: N"` where N is greater than 0, or run `ollama list` from the service's shell if Railway exposes one. `total blobs: 0` means the model isn't there and requests will fail.
