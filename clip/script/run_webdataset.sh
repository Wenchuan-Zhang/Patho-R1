#!/bin/bash

# WebDataset Creation Script
# This script runs the create_webdataset.py with configurable parameters

set -e  # Exit on any error

# =============================================================================
# Configuration - Modify these paths and parameters as needed
# =============================================================================

# Input paths
CSV_PATH="/path/to/caption.csv"
IMAGE_DIR="/path/to/images"
OUTPUT_DIR="/path/to/output"

# Processing parameters
SAMPLES_PER_SHARD=5000
NUM_WORKERS=8
START_SHARD_IDX=324

# Optional: Enable verbose logging
VERBOSE=false

# =============================================================================
# Script execution
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python packages
check_python_packages() {
    local packages=("pandas" "Pillow" "tqdm")
    local missing_packages=()
    
    for package in "${packages[@]}"; do
        if ! python3 -c "import $package" >/dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_error "Missing Python packages: ${missing_packages[*]}"
        print_info "Install them with: pip install ${missing_packages[*]}"
        return 1
    fi
    
    return 0
}

# Function to validate paths
validate_paths() {
    local errors=0
    
    if [[ ! -f "$CSV_PATH" ]]; then
        print_error "CSV file not found: $CSV_PATH"
        errors=$((errors + 1))
    fi
    
    if [[ ! -d "$IMAGE_DIR" ]]; then
        print_error "Image directory not found: $IMAGE_DIR"
        errors=$((errors + 1))
    fi
    
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        print_warning "Output directory doesn't exist, will be created: $OUTPUT_DIR"
    fi
    
    return $errors
}

# Function to estimate disk space
estimate_disk_space() {
    if [[ -f "$CSV_PATH" ]]; then
        local csv_lines=$(wc -l < "$CSV_PATH")
        local estimated_shards=$(( (csv_lines - 1 + SAMPLES_PER_SHARD - 1) / SAMPLES_PER_SHARD ))
        print_info "Estimated number of shards: $estimated_shards"
        print_info "Make sure you have sufficient disk space in: $OUTPUT_DIR"
    fi
}

# Main execution function
main() {
    print_info "Starting WebDataset creation process..."
    print_info "Timestamp: $(date)"
    
    # Check if Python 3 is available
    if ! command_exists python3; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if create_webdataset.py exists
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local python_script="$script_dir/create_webdataset.py"
    
    if [[ ! -f "$python_script" ]]; then
        print_error "Python script not found: $python_script"
        print_info "Make sure create_webdataset.py is in the same directory as this script"
        exit 1
    fi
    
    # Check Python packages
    print_info "Checking Python dependencies..."
    if ! check_python_packages; then
        exit 1
    fi
    print_success "All Python dependencies are available"
    
    # Validate paths
    print_info "Validating input paths..."
    if ! validate_paths; then
        exit 1
    fi
    print_success "Path validation completed"
    
    # Show configuration
    print_info "Configuration:"
    echo "  CSV Path: $CSV_PATH"
    echo "  Image Directory: $IMAGE_DIR"
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  Samples per shard: $SAMPLES_PER_SHARD"
    echo "  Number of workers: $NUM_WORKERS"
    echo "  Starting shard index: $START_SHARD_IDX"
    echo "  Verbose logging: $VERBOSE"
    
    # Estimate disk space
    estimate_disk_space
    
    # Ask for confirmation
    echo
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled by user"
        exit 0
    fi
    
    # Build command arguments
    local cmd_args=(
        "--csv" "$CSV_PATH"
        "--images" "$IMAGE_DIR"
        "--output" "$OUTPUT_DIR"
        "--samples-per-shard" "$SAMPLES_PER_SHARD"
        "--workers" "$NUM_WORKERS"
        "--start-shard-idx" "$START_SHARD_IDX"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd_args+=("--verbose")
    fi
    
    # Execute the Python script
    print_info "Starting WebDataset creation..."
    print_info "Command: python3 $python_script ${cmd_args[*]}"
    
    local start_time=$(date +%s)
    
    if python3 "$python_script" "${cmd_args[@]}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "WebDataset creation completed successfully!"
        print_info "Total time: ${duration} seconds"
    else
        local exit_code=$?
        print_error "WebDataset creation failed with exit code: $exit_code"
        exit $exit_code
    fi
}

# Function to show help
show_help() {
    echo "WebDataset Creation Script"
    echo
    echo "This script creates WebDataset tar shards from a CSV file and image directory."
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Enable verbose logging"
    echo "  --dry-run      Show configuration without running"
    echo
    echo "Configuration is done by editing the variables at the top of this script."
    echo
    echo "Required directory structure:"
    echo "  - CSV file with 'image_path' and 'caption' columns"
    echo "  - Image directory containing the images referenced in CSV"
    echo "  - Output directory (will be created if it doesn't exist)"
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            print_info "Dry run mode - showing configuration:"
            echo "  CSV Path: $CSV_PATH"
            echo "  Image Directory: $IMAGE_DIR"
            echo "  Output Directory: $OUTPUT_DIR"
            echo "  Samples per shard: $SAMPLES_PER_SHARD"
            echo "  Number of workers: $NUM_WORKERS"
            echo "  Starting shard index: $START_SHARD_IDX"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main