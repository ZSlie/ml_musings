#!/bin/bash

dataset_path="${HOME}/datasets/TradingData"

# create the header column
for folder in automotive financial; do
    touch $dataset_path/$folder/stockquotes_combined.csv
    tail -q $dataset_path/$folder/*stockquotes*.csv >> $dataset_path/$folder/stockquotes_combined.csv
    touch $dataset_path/$folder/options_combined.csv
    tail -q $dataset_path/$folder/*options_*.csv >> $dataset_path/$folder/options_combined.csv
    touch $dataset_path/$folder/optionstats_combined.csv
    tail -q $dataset_path/$folder/*optionstats_*.csv >> $dataset_path/$folder/optionstats_combined.csv
done

wait