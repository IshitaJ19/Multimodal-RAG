#!/bin/sh

# Directory where Ollama stores models
MODEL_DIR="${OLLAMA_MODELS:-/root/.ollama/models}"

# Array of model names
MODELS="llama3.2 gemma3:12b"

for MODEL in $MODELS; do
  # Convert model name to expected filename format
  MODEL_PATH="$MODEL_DIR/$(echo "$MODEL" | sed 's/:/_/').bin"

  if [ -f "$MODEL_PATH" ]; then
    echo "Model $MODEL already exists at $MODEL_PATH"
  else
    echo "Model $MODEL not found. Pulling..."
    ollama pull "$MODEL"
  fi
done
