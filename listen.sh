#!/bin/bash
ID=$1
echo "=== CLEAN ==="
afplay "data/processed/test/${ID}_clean.wav"
echo "=== NOISY ==="
afplay "data/processed/test/${ID}_noisy.wav"
echo "=== ENHANCED ==="
afplay "results/finetuned_v2/${ID}_enhanced.wav"
