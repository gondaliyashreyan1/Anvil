#!/bin/bash
# Benchmark: anvil vs llama.cpp turboquant

MODEL="$HOME/.anvil/models/Bonsai-27B-Q1_0.gguf"
LLAMA_BENCH="/Users/shreyangondaliya/Downloads/turboquant-plus-tqp-v0.1.1/llama-bench"
LLAMA_CLI="/Users/shreyangondaliya/Downloads/turboquant-plus-tqp-v0.1.1/llama-cli"
ANVIL="./build/anvil"

echo "=== llama-bench turbo3 ==="
$LLAMA_BENCH -m "$MODEL" -ngl 99 -t 10 -fa 1 -ctk turbo3 -ctv turbo3 -p 128 -n 128 -r 3 2>&1

echo ""
echo "=== llama-bench f16 ==="
$LLAMA_BENCH -m "$MODEL" -ngl 99 -t 10 -fa 1 -p 128 -n 128 -r 3 2>&1

echo ""
echo "=== anvil turbo3 ==="
echo "What is 2+2? Answer with just the number." | $ANVIL "$MODEL" --ctx 262144 --cache-type-k turbo3 --cache-type-v turbo3 --flash-attn --no-tui --ngl 99 2>/dev/null

echo ""
echo "=== llama-cli turbo3 ==="
$LLAMA_CLI -m "$MODEL" -ngl 99 -t 10 -fa 1 -c 262144 --cache-type-k turbo3 --cache-type-v turbo3 -p "What is 2+2? Answer with just the number." 2>/dev/null
