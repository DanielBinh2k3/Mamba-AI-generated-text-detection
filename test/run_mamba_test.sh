#!/bin/bash

# Set default parameters
MODEL_NAME="state-spaces/mamba2-1.3b"
MODEL_PATH="./Models/Mamba2Former-Accuracy-97.36-Loss-220.pth"
SEED=0
DEVICE="cuda"

# Define the directories
DATA_DIR="./exp_test/data/"
RESULTS_DIR="./exp_test/results/"

# Ensure results directory exists
mkdir -p $RESULTS_DIR

# Iterate over dataset files in the DATA_DIR
for DATASET_FILE in $DATA_DIR*.raw_data.json; do
  BASE_NAME_DATA=$(basename "$DATASET_FILE")
  BASE_NAME_MODEL=$(basename "$MODEL_PATH")
  # Define the output file path based on the base name
  mkdir -p "${RESULTS_DIR}${BASE_NAME_MODEL}"
  OUTPUT_FILE="${RESULTS_DIR}${BASE_NAME_MODEL}/${BASE_NAME_DATA}.results.json"
  # Run the Python script with the parameters
  python3 mamba_test.py \
    --output_file "$OUTPUT_FILE" \
    --dataset_file "$DATASET_FILE" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed "$SEED" \
    --device "$DEVICE"


  echo "Completed processing: $DATASET_FILE, results saved to $OUTPUT_FILE"
done
