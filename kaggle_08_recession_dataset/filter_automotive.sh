#!/bin/bash

dataset_path="${HOME}/datasets/TradingData"
input_directory="$dataset_path/unzipped"
output_directory="$dataset_path/automotive"
mkdir -p $output_directory
# DDAIF - owner of Chrysler pior to 2007 acquisition by Cerberus
# GM - general motors

for file in "$input_directory"/*.csv; do
    awk -F, '$1 == "DDIAF" || $1 == "GM"' "$file" > "$output_file/$(basename "$file")" &
done

wait