# Standalone Railway service running Ollama. The model is pulled at
# container startup (start.sh), not baked into the image at build time —
# baking it in via a backgrounded-server build trick turned out unreliable
# in practice (the image shipped with zero blobs despite the build step
# appearing to succeed). Attach a Railway Volume at /root/.ollama so the
# pulled model persists across restarts/redeploys — see README.md.
FROM ollama/ollama:latest

# Override with a Railway build arg if you want a different model.
# The earlier 100s timeouts (14b, then 7b, then 3b, all hitting the same
# wall) turned out to be cold-start: the first generate call after a
# container boots pays for paging the model weights into RAM, competing
# with REQUEST_TIMEOUT_MS in lib/ollama.ts. Once warm, llama3.2:3b answered
# reliably every time. qwen2.5:7b-instruct is the next step up now that
# warm-state performance is confirmed — a real quality jump over 3b, still
# small enough to fail fast if it doesn't hold up.
ARG MODEL=qwen2.5:7b-instruct
ENV MODEL_NAME=${MODEL}
ENV OLLAMA_HOST=0.0.0.0:11434

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 11434
ENTRYPOINT ["/start.sh"]
