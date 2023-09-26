#!/bin/bash

dataset_path="${HOME}/datasets/TradingData"
input_directory="$dataset_path/unzipped"
output_directory="$dataset_path/financial"
mkdir -p $output_directory
# BAC - Bank of America
# C - Citigroup
# GS - Goldman Sachs
# JPM - JP Morgan
# MS - Morgan Stanley
# WFC - Wells Fargo
# BK - Bank of NY Mellon Corp
# STT - State Street Bank


for file in "$input_directory"/*.csv; do
    # output_file="$output_directory/$(basename "$file")" # don't use since loops are globally scoped
    echo $output_file
    awk -F, '$1 == "BAC" || $1 == "C" || $1 == "GS" || $1 == "JPM" || $1 == "MS" || $1 == "WFC" || $1 == "BK" || $1 == "STT"' "$file" > "$output_directory/$(basename "$file")" &
done

wait 