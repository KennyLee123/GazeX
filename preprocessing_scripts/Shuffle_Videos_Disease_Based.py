import os
import cv2
import numpy as np
import csv
from pathlib import Path

def create_shuffle_directory(input_path, output_base_dir):
    # Parse input path
    input_path = Path(input_path)
    relative_path = input_path.relative_to(input_path.parents[3])  # Adjust to keep structure from data_disease_video
    output_path = Path(output_base_dir) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def shuffle_video_frames(input_video_path, output_video_path, csv_output_path=None):
    # Open the video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video file: {input_video_path}")
        return False, None, None

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"Warning: No frames found in video: {input_video_path}")
        return False, None, None

    # Create shuffle index
    original_indices = list(range(len(frames)))
    if len(frames)==1:
        return False, None, None
    shuffled_indices = original_indices.copy()
    np.random.shuffle(shuffled_indices)

    # Write shuffled video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Warning: Could not create output video file: {output_video_path}")
        return False, None, None

    for idx in shuffled_indices:
        out.write(frames[idx])
    out.release()

    # Write individual CSV with sequence if csv_output_path is provided
    if csv_output_path:
        with open(csv_output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(shuffled_indices)  # Write single row of original indices

    # Return sequence as a comma-separated string
    sequence_str = ','.join(map(str, shuffled_indices))
    return True, csv_output_path, sequence_str

def process_all_videos(input_dir, output_base_dir):
    input_dir = Path(input_dir)
    output_base_dir = Path(output_base_dir)
    csv_paths_file = output_base_dir / "video_csv_paths.csv"

    # List to store video paths, CSV paths, and sequences
    path_records = []

    # Walk through all subdirectories
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                input_video_path = Path(root) / file
                output_video_path = create_shuffle_directory(input_video_path, output_base_dir)
                csv_output_path = output_video_path.with_suffix('.csv')

                print(f"Processing: {input_video_path}")
                success, csv_path, sequence = shuffle_video_frames(input_video_path, output_video_path, csv_output_path)
                if success:
                    print(f"Shuffled video saved to: {output_video_path}")
                    if csv_path:
                        print(f"Frame sequence CSV saved to: {csv_output_path}")
                    # Store relative paths and sequence
                    relative_video_path = output_video_path.relative_to(output_base_dir)
                    relative_csv_path = csv_output_path.relative_to(output_base_dir) if csv_path else ''
                    path_records.append([str(relative_video_path), str(relative_csv_path), sequence])
                else:
                    print(f"Failed to process: {input_video_path}")

    # Write CSV with video paths, CSV paths, and sequences
    with open(csv_paths_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video_path', 'csv_path', 'frame_sequence'])
        writer.writerows(path_records)

    print(f"Video paths and sequences saved to: {csv_paths_file}")

def main():
    input_dir = "./data_disease_video"
    output_base_dir = "./data_disease_video_shuffle"

    process_all_videos(input_dir, output_base_dir)
    print("All videos processed.")

if __name__ == "__main__":
    main()
