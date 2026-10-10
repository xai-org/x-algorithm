#!/bin/sh
set -e

ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama server to accept connections..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

if ollama list | grep -q "^${MODEL_NAME}"; then
  echo "Model ${MODEL_NAME} already present, skipping pull."
else
  echo "Pulling model ${MODEL_NAME} (first boot only if a volume is attached — this can take several minutes)..."
  ollama pull "${MODEL_NAME}"
fi

# Weights aren't in RAM just because they're on disk — the first
# /api/generate call after boot pays to page them in, and that cost was
# what earlier looked like a hard timeout wall (see README). Eating that
# cost here, once, at startup means the first real user request lands on
# an already-warm model instead of racing the app's request timeout.
echo "Warming up ${MODEL_NAME}..."
ollama run "${MODEL_NAME}" "hi" >/dev/null 2>&1 || echo "Warmup call failed, continuing anyway — model will warm on first real request instead."

wait $SERVER_PID
