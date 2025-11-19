import pandas as pd
import cv2
import os
import ast
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_frames_to_video(input_video_path, output_video_path, frame_indices, fps=1):
    """
    Extract specific frames from input video and save as new video
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to save output video
        frame_indices: List of frame indices to extract
        fps: Frames per second for output video
    """
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory
    create_directory(os.path.dirname(output_video_path))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Extract specified frames
    frames_extracted = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in frame_indices:
            frames_extracted.append(frame)
            
        frame_count += 1
    
    # Write extracted frames to output video
    for frame in frames_extracted:
        out.write(frame)
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"Extracted {len(frames_extracted)} frames to {output_video_path}")
    return True

def process_csv_and_extract_videos(csv_path):
    """
    Process CSV file and extract videos based on frame indices and disease categories
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Process each row
    for index, row in df.iterrows():
        patient_id = row['id']
        frame_index_str = row['frame_index']
        patch_sentences_str = row['patch_sentences']
        
        print(f"\nProcessing row {index + 1}: {patient_id}")
        
        # Parse frame_index and patch_sentences
        try:
            frame_indices = ast.literal_eval(frame_index_str)
            patch_sentences = ast.literal_eval(patch_sentences_str)
        except Exception as e:
            print(f"Error parsing data for {patient_id}: {e}")
            continue
        
        # Input video path
        input_video_path = f"./data/{patient_id}.mp4"
        
        # Check if input video exists
        if not os.path.exists(input_video_path):
            print(f"Warning: Video file not found: {input_video_path}")
            continue
        
        # Calculate frame ranges based on cumulative indices
        frame_start = 0
        
        # Process each disease category
        for i, (sentence, cumulative_end) in enumerate(zip(patch_sentences, frame_indices)):
            # Clean sentence for directory name (remove special characters)
            disease_name = sentence.strip().replace(' .', '').replace('/', '_').replace('\\', '_')
            
            # Calculate frame range for this disease
            frame_end = cumulative_end
            frames_to_extract = list(range(frame_start, frame_end))
            
            print(f"  Disease {i+1}: {disease_name}")
            print(f"  Cumulative end: {cumulative_end}")
            print(f"  Frame range: {frame_start} to {frame_end-1} (total: {frame_end - frame_start} frames)")
            
            # Update frame_start for next disease
            frame_start = frame_end
            
            # Output video path
            output_dir = f"./data_disease_video_phase_2/{disease_name}"
            output_video_path = f"{output_dir}/{patient_id}.mp4"
            
            print(f"  Disease: {disease_name}")
            print(f"  Extracting frames: {frames_to_extract}")
            
            print(f"  Output: {output_video_path}")
            
            # Extract frames and create video
            success = extract_frames_to_video(
                input_video_path, 
                output_video_path, 
                frames_to_extract
            )
            
            if not success:
                print(f"  Failed to extract video for {disease_name}")
                return
        # break

def main():
    """Main function"""
    csv_path = "./patch_metadata_enhanced_2.csv"
    
    print("Starting video frame extraction process...")
    print(f"CSV file: {csv_path}")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Process CSV and extract videos
    process_csv_and_extract_videos(csv_path)
    
    print("\nVideo extraction process completed!")

if __name__ == "__main__":
    main()