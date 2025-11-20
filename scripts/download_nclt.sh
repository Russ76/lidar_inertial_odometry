#!/bin/bash
# NCLT Dataset Download Script
# University of Michigan North Campus Long-Term Vision and LiDAR Dataset
# http://robots.engin.umich.edu/nclt/

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
DATA_DIR="${1:-$PROJECT_ROOT/benchmark/datasets/NCLT}"
SEQUENCES=("2012-01-08" "2012-01-15" "2012-01-22" "2012-02-02" "2012-02-04" "2012-02-05" "2012-02-12" "2012-02-18" "2012-02-19")

# NCLT sequences used in Faster-LIO paper (need to figure out which dates correspond to nclt_2 and nclt_4)
# The dataset has multiple dates, and we need to download the ones used in the paper

echo "=========================================="
echo "NCLT Dataset Downloader"
echo "=========================================="
echo "Download directory: $DATA_DIR"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# NCLT Base URL
BASE_URL="http://robots.engin.umich.edu/nclt"

# Function to download a sequence
download_sequence() {
    local date=$1
    local seq_dir="$DATA_DIR/$date"
    
    echo "=========================================="
    echo "Downloading sequence: $date"
    echo "=========================================="
    
    mkdir -p "$seq_dir"
    cd "$seq_dir"
    
    # Download LiDAR data (Velodyne HDL-32E)
    echo "Downloading LiDAR data..."
    if [ ! -f "velodyne_sync.tar.gz" ]; then
        wget -c "$BASE_URL/$date/velodyne_sync.tar.gz" || echo "Failed to download LiDAR data"
        tar -xzf "velodyne_sync.tar.gz"
    else
        echo "LiDAR data already exists, skipping..."
    fi
    
    # Download IMU data (Microstrain 3DM-GX3-25)
    echo "Downloading IMU data..."
    if [ ! -f "ms25.tar.gz" ]; then
        wget -c "$BASE_URL/$date/ms25.tar.gz" || echo "Failed to download IMU data"
        tar -xzf "ms25.tar.gz"
    else
        echo "IMU data already exists, skipping..."
    fi
    
    # Download Ground Truth (GPS/RTK)
    echo "Downloading Ground Truth..."
    if [ ! -f "ground_truth.tar.gz" ]; then
        wget -c "$BASE_URL/$date/ground_truth.tar.gz" || echo "Failed to download ground truth"
        tar -xzf "ground_truth.tar.gz"
    else
        echo "Ground truth already exists, skipping..."
    fi
    
    echo "Completed: $date"
    echo ""
}

# Print menu
echo "Available sequences:"
echo "  1) 2012-01-08 (Small campus loop)"
echo "  2) 2012-01-15"
echo "  3) 2012-01-22"
echo "  4) 2012-02-02"
echo "  5) 2012-02-04"
echo "  6) 2012-02-05"
echo "  7) 2012-02-12"
echo "  8) 2012-02-18"
echo "  9) 2012-02-19"
echo "  a) Download ALL (Warning: Very large! ~500GB total)"
echo "  s) Download SMALL set (2012-01-08, 2012-02-04) - Recommended for testing"
echo ""
read -p "Select sequences to download (e.g., '1 4' or 'a' or 's'): " selection

case $selection in
    a|A)
        echo "Downloading ALL sequences..."
        for seq in "${SEQUENCES[@]}"; do
            download_sequence "$seq"
        done
        ;;
    s|S)
        echo "Downloading SMALL test set..."
        download_sequence "2012-01-08"
        download_sequence "2012-02-04"
        ;;
    *)
        for num in $selection; do
            case $num in
                1) download_sequence "2012-01-08" ;;
                2) download_sequence "2012-01-15" ;;
                3) download_sequence "2012-01-22" ;;
                4) download_sequence "2012-02-02" ;;
                5) download_sequence "2012-02-04" ;;
                6) download_sequence "2012-02-05" ;;
                7) download_sequence "2012-02-12" ;;
                8) download_sequence "2012-02-18" ;;
                9) download_sequence "2012-02-19" ;;
                *) echo "Invalid selection: $num" ;;
            esac
        done
        ;;
esac

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Data location: $DATA_DIR"
echo ""
echo "To use with lio_player:"
echo "  ./lio_player ../config/nclt.yaml $DATA_DIR/2012-01-08"
echo ""
