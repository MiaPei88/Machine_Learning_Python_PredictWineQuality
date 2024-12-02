#!/bin/bash

# Define the Python script and the CSV file as variables
PYTHON_SCRIPT="wine_quality_pred.py"
CSV_FILE="winequality-white.csv"

# Execute the Python script with the CSV file as input
echo "Running Python script '$PYTHON_SCRIPT' with input file '$CSV_FILE'..."
python3 "$PYTHON_SCRIPT" "$CSV_FILE" ./

