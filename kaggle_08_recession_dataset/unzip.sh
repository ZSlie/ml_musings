#!/bin/bash

# Grant executable permission: chmod -R +x unzip.sh
dataset_path="${HOME}/datasets/TradingData"
source_path="$dataset_path/zipped"
destination_path="$dataset_path/unzipped"

# unzip
echo "Unzipping all files in $source_path"

unzip $source_path/\*.zip -d $destination_path