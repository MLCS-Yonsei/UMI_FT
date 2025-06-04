#!/bin/bash
# Set your demos directory here:
DEMOS_DIR=~/GoPro_20250523/demos

# Function to check if a folder has exactly 10 files (only regular files)
check_folder_success() {
    folder="$1"
    count=$(find "$folder" -maxdepth 1 -type f | wc -l)
    # Remove any leading/trailing whitespace from count
    count=$(echo "$count" | xargs)
    if [ "$count" -eq 10 ]; then
        return 0
    else
        return 1
    fi
}

while true; do
    echo "Running batch SLAM on demos in: $DEMOS_DIR"
    python scripts_slam_pipeline/03_batch_slam.py -i "$DEMOS_DIR"
    
    total_demos=0
    fail_count=0
    echo "Checking demo folders..."
    for demo_folder in "$DEMOS_DIR"/demo*; do
        if [ -d "$demo_folder" ]; then
            total_demos=$(( total_demos + 1 ))
            if check_folder_success "$demo_folder"; then
                echo "[$(basename "$demo_folder")] Success: 10 files found."
            else
                count=$(find "$demo_folder" -maxdepth 1 -type f | wc -l | xargs)
                echo "[$(basename "$demo_folder")] FAIL: Found $count files (expected 10)."
                fail_count=$(( fail_count + 1 ))
            fi
        fi
    done

    # Calculate allowed failures: allow up to 10% of demos to fail.
    # We round up by adding (total_demos + 9) / 10.
    # allowed_fail=$(( (total_demos + 9) / 10 ))
    allowed_fail=1
    echo "Total demos: $total_demos, Failed demos: $fail_count, Allowed failures: $allowed_fail"

    if [ "$fail_count" -le "$allowed_fail" ]; then
        echo "Failure rate within threshold (≤10% failures). Exiting."
        break
    else
        echo "Too many demos failed. Retrying in 5 seconds..."
        sleep 5
    fi
done

python run_slam_pipeline_ft.py ~/GoPro_20250523
echo "Waiting for dataset_plan.pkl to appear in ~/GoPro_20250523..."
while [ ! -f ~/GoPro_20250523/dataset_plan.pkl ]; do
    sleep 5
done

echo "Running generate replay buffer..."
python scripts_slam_pipeline/07_generate_replay_buffer_ft.py -o ~/GoPro_20250523/dataset.zarr.zip ~/GoPro_20250523
while [ ! -f ~/GoPro_20250523/dataset.zarr.zip ]; do
    sleep 5
done

echo "All commands completed."