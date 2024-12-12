#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download the file
wget 'https://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr20.fa.gz' -O ./data/chr20.fa.gz
if [ $? -ne 0 ]; then
    echo "Failed to download chr20.fa.gz"
    exit 1
fi

# Unzip the file
gunzip ./data/chr20.fa.gz
if [ $? -ne 0 ]; then
    echo "Failed to unzip chr20.fa.gz"
    exit 1
fi

fasta_dir="data"
data_dir="files_to_be_compressed"
mkdir -p $data_dir

# Process each .fa file
for f in $fasta_dir/*.fa; do
    if [ ! -f "$f" ]; then
        echo "No .fa files found in $fasta_dir"
        exit 1
    fi

    echo "filename: $f"
    s=${f##*/}
    basename=${s%.*}
    echo $basename

    output_file="$data_dir/$basename.txt"
    sed '/>/d' $f | tr -d '\n' | tr '[:lower:]' '[:upper:]' > $output_file
    echo "- - - - - "
done