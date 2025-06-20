#!/bin/bash

# Run model evaluation

# Check if MUSDB exists
if [ ! -d "./data/musdb18" ]; then
    echo "MUSDB dataset not found in ./data/musdb18"
    echo "Please download and extract MUSDB18 dataset first"
    exit 1
fi

# Check if checkpoint exists
CHECKPOINT="checkpoints/best_model.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Model checkpoint not found at $CHECKPOINT"
    echo "Please train the model first or specify a different checkpoint path"
    exit 1
fi

# Create evaluation directory
mkdir -p evaluation_results

# Run evaluation
echo "Starting model evaluation..."
python -m evaluation.evaluate_model --model_path "$CHECKPOINT"

# Generate visualizations
echo "Generating visualizations..."
python -m evaluation.visualize_results

echo "Evaluation complete! Results saved in evaluation_results/"