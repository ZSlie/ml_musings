#!/bin/bash

# start with uploading IAM Handwriting top50 to cloud.google.com/storage/pricing
# best to do this with "folder" upload rather than individual files

# configure the G Cloud CLI
hw_dir="${HOME}/datasets/handwriting/IAM_Handwriting_Top50"
src_dir="$hw_dir/data_subset"
dest_dir="$hw_dir/gcloud_analysis"

for file in $src_dir/*.png; do
    file_bn=$(basename $file)
    fn_no_ext=$(basename $file .png)

    src="gs://iam_handwriting_top50/$file_bn"
    dest="$dest_dir/$fn_no_ext.json"
    echo "Processing: $file_bn"
    # gcloud ml vision detect-document $src > $dest # uncomment as needed.
done