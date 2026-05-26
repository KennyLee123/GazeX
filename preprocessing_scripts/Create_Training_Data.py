import pandas as pd
import ast
import os
import csv
from pathlib import Path
from sklearn.utils import shuffle
matching_df = pd.read_csv("./all_bboxes_and_centroids.csv")
def find_matching_row(sentence):
    # Find rows where description matches
    matching_rows = matching_df[matching_df['sentence'].str.strip().str.strip('.').str.strip() == sentence.strip().strip('.').strip()]
    if not matching_rows.empty:
        row = matching_rows.iloc[0]  # Use iloc to get the first matching row
        return row["reasoning"]
    return None

# Example usage
# input_sentence = "a calcified granuloma is present in left lung apex."  # Replace with your sentence
# result = find_matching_row(input_sentence)
def read_transcription(file_path):
    try:
        # Check if file exists
        file_path = Path(file_path)
        if not file_path.exists():
            # print(f"File not found: {file_path}")
            return None
        
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # print("Transcription content:")
        # print(content)
        return content
    
    except Exception as e:
        # print(f"Error reading file: {e}")
        return None
def gen_seq():
    csv_path = "./data_shuffle_global/video_csv_paths.csv"
    shuffled_root = "./data_shuffle_global"
    original_image_root = "./image_jpg"

    dataset = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence_name = row["frame_sequence"]
            video_path = row['video_path']

            folder_path = os.path.join(shuffled_root, video_path)

            base_name = video_path.replace(".mp4", "")
            original_image_path = os.path.join(original_image_root, f"{base_name}.jpg")
            if not os.path.isfile(original_image_path):
                continue  # skip if original image is missing

            # Build prompt

            transcript = read_transcription(f"../reflacx-xray-localization/1.0.0/main_data/{base_name}/transcription.txt")

            dataset.append( {
            "messages": [
                {
                    "content": f"<video>Given a shuffled radiologist's eye-tracking video with {len(sequence_name.split(','))} frames, which captures how a radiologist examines an X-ray image, analyze the video to identify the correct sequence of frames and the specific regions of the X-ray attended by the radiologist's gaze in each frame. <image> Using the provided X-ray image and knowledge of a radiologist's eye-tracking pattern, first generate the accurate sequence of frames for the eye-tracking video. Then, provide a detailed description of the X-ray image, ensuring the number of descriptions matches the number of frames in the eye-tracking video. Provide your answer in the following format <seq>32145</seq><description>description1,description2,...</descriptions>.",
                    "role": "user"
                },
                {
                    "content":f"<seq>{''.join(sequence_name.split(','))}</seq><description>{transcript}</descriptions>",  # List of descriptions for each frame
                    "role": "assistant"
                },
                

            ],
            "videos": [str(folder_path)],
            "images": [str(original_image_path)]
            })
    return dataset


def gen_seq_localized():
    csv_path = "./data_disease_video_shuffle/video_csv_paths.csv"

    dataset = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence_name = row["frame_sequence"]
            video_path = row['video_path']
            dataset.append( {
            "messages": [
                {
                    "content": f"<video>Given a shuffled radiologist's eye-tracking video with {len(sequence_name.split(','))} frames, which captures how a radiologist examines an X-ray image for description '{str(video_path).split('/')[-2]}', analyze the video to identify the correct sequence of gazing to reach the diagnosis. Provide your answer in the following format <seq>32145</seq>, where the first index number you provided should be the frame to be analyzed first.",
                    "role": "user"
                },
                {
                    "content":f"<seq>{''.join(sequence_name.split(','))}</seq>",  # List of descriptions for each frame
                    "role": "assistant"
                },
            ],
            "videos": ['./data_disease_video_shuffle/'+str(video_path)],
            })
            # print(dataset)
            # return
    return dataset

result = [] 
for data_csv in ['patch_metadata_enhanced_3_train','patch_metadata_enhanced_1']:
  data = pd.read_csv(f"./{data_csv}.csv")
  for index,row in data.iterrows():
      frame_index_str = row['frame_index']
      patch_sentences_str = row['patch_sentences']
      

      frame_indices = ast.literal_eval(frame_index_str)
      patch_sentences = ast.literal_eval(patch_sentences_str)
      # print(row['id'])
      result.append({
      "messages": [
        {
          "content": f"""<video>This video contains {frame_indices[-1]} frames of a radiologist's eye-tracking data while examining an X-ray image. Each frame shows where the radiologist focused their gaze at that point in time. <image> This is the corresponding X-ray image. Using the gaze patterns and the image, generate a detailed description or diagnosis that reflects the radiologist's interpretation, guided by their attention over time.""",
          "role": "user"
        },
        {
          "content": f"{' '.join(patch_sentences)}",
          "role": "assistant"
        }
      ],
      "videos": [
        f"./data/{row['id']}.mp4"
      ],
      "images": [
        f"./image_jpg/{row['id']}.jpg"
      ]
     })
      result.append({
      "messages": [
        {
          "content": f"""<image> Given an X-ray image, using the radiologist's interpretation pattern, generate a detailed description or diagnosis for the image.""",
          "role": "user"
        },
        {
          "content": f"{' '.join(patch_sentences)}",
          "role": "assistant"
        }
      ],
      "images": [
        f"./image_jpg/{row['id']}.jpg"
      ]
     })

root_dir = "./data_disease_video"
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    # Check if the item is a directory
    if os.path.isdir(subdir_path):
        # print(f"\nSubdirectory: {subdir}")
        # Loop through files in the subdirectory
        for file in os.listdir(subdir_path):
            # Check if the file has .mp4 extension (case-insensitive)
            if file.lower().endswith(".mp4"):
              result.append({
              "messages": [
                {
                  "content": f"""<video>This video contains a radiologist's gazing data while examining an X-ray image <image>. Each frame shows where the radiologist focused their gaze at that point in time. Generate a description that reflects the radiologist's interpretation, guided by their attention over time.""",
                  "role": "user"
                },
                {
                  "content": f"{subdir}",
                  "role": "assistant"
                }
              ],
              "images": [
                    f"./image_jpg/{file.split('.')[0]}.jpg"
                ],
              "videos": [
                f"{os.path.join(subdir_path,file)}"
              ],
            })

# # Group patches by patch_id
from collections import defaultdict

def narrow_bbox_by_percentage(xmin, ymin, xmax, ymax, percentage=5):
    """
    Narrow a bounding box by a given percentage equally on all sides.
    
    Args:
        xmin, ymin, xmax, ymax: Original bounding box coordinates
        percentage: Percentage to narrow by (default 5%)
    
    Returns:
        tuple: New (xmin, ymin, xmax, ymax) coordinates
    """
    if xmin>xmax:
      print('fuck')
    if ymin>ymax:
      print('fuck')

    # Calculate width and height
    width = xmax - xmin
    height = ymax - ymin
    
    # Calculate the amount to narrow by (as a fraction)
    narrow_fraction = percentage / 100
    
    # Calculate the reduction amount for each dimension
    width_reduction = width * narrow_fraction
    height_reduction = height * narrow_fraction
    
    # Apply the reduction equally to all sides
    new_xmin = xmin + (width_reduction / 2)
    new_ymin = ymin + (height_reduction / 2)
    new_xmax = xmax - (width_reduction / 2)
    new_ymax = ymax - (height_reduction / 2)
    
    return new_xmin, new_ymin, new_xmax, new_ymax

for data_csv in ['patch_metadata_enhanced_bbox_1','patch_metadata_enhanced_bbox_3_train']:
  data = pd.read_csv(f"./{data_csv}.csv")
  for index,row in data.iterrows():
      bboxes_seq = ast.literal_eval(row['bboxes_seq'])

      grouped_patches = defaultdict(list)
      for patch in bboxes_seq:
          grouped_patches[patch['patch_id']].append(patch)
      for patch_id in sorted(grouped_patches.keys()):
        temp_sentence = ""
        temp_box_sentence  = ""
        formatted_bbox_clues =f"visual clues : ["
        for patch in grouped_patches[patch_id]:
            # print(f"  Cluster ID: {patch['cluster_id']}, Sentence: {patch['sentence']}, BBox: {patch['bbox']}")
            temp_sentence = patch['sentence']
            point0, point1, point2, point3, point4, point5 = patch['bbox']
            if patch['cluster_id'] != 'holistic_end':
              if patch['cluster_id'] !=0:
                formatted_bbox_clues += ','
              point0, point1, point2, point3 = map(int, narrow_bbox_by_percentage(point0, point1, point2, point3, 0.35))
              formatted_bbox_clues += f"({point0},{point1})({point2},{point3})({int(point4)},{int(point5)})"
              # temp_box.append(formatted_bbox)
            else:
              formatted_bbox_clues += '] '
              point0, point1, point2, point3 = map(int, narrow_bbox_by_percentage(point0, point1, point2, point3, 0.15))
              formatted_bbox = f"observation : [({point0},{point1})({point2},{point3})({int(point4)},{int(point5)})]"
              final_observation_area = f"[({point0},{point1})({point2},{point3})({int(point4)},{int(point5)})]"
              # temp_box.append(formatted_bbox)
        if formatted_bbox_clues!=f"visual clues : [] ":
          temp_box_sentence+=formatted_bbox_clues
          temp_box_sentence+=formatted_bbox
          result.append({
                "messages": [
                {
                    "content": f"""<image>Given an X-ray image and a radiologist's description '{temp_sentence}', mimic a radiologist's interpretation pattern to identify visual clues and an overall observation area. Analyze the image to detect relevant features (e.g., abnormal opacities, fractures, lesions, or other radiological findings) that correspond to the provided description. Output as much visual clues to support your observation and generate only one observation area in the format of bounding boxes and centroids, using the structure: 'visual clues: [(xmin1, ymin1)(xmax1, ymax1)(centroidx1, centroidy1), (xmin2, ymin2)(xmax2, ymax2)(centroidx2, centroidy2), ...], observation: [(xmin, ymin)(xmax, ymax)(centroidx, centroidy)]'.""",
                    "role": "user"
                },
                {
                    "content": f"{temp_box_sentence}",
                    "role": "assistant"
                }
                ],
                "images": [
                f"./image_jpg/{row['id']}.jpg"
                ],
            })
        else:
          temp_box_sentence+=formatted_bbox
          result.append({
                "messages": [
                {
                    "content": f"""<image>Given an X-ray image and a radiologist's description '{temp_sentence}', mimic a radiologist's interpretation pattern to identify visual clues and an overall observation area. Analyze the image to detect relevant features (e.g., abnormal opacities, fractures, lesions, or other radiological findings) that correspond to the provided description. Output visual clues and observation area in the format of bounding boxes and centroids, using the structure: 'visual clues: [(xmin1, ymin1)(xmax1, ymax1)(centroidx1, centroidy1), (xmin2, ymin2)(xmax2, ymax2)(centroidx2, centroidy2), ...], observation: [(xmin, ymin)(xmax, ymax)(centroidx, centroidy)]'.""",
                    "role": "user"
                },
                {
                    "content": f"{temp_box_sentence}",
                    "role": "assistant"
                }
                ],
                "images": [
                f"./image_jpg/{row['id']}.jpg"
                ],
            })
        # matching_row = find_matching_row(temp_sentence)
        # if matching_row!=None:
        #   result.append({
        #           "messages": [
        #             {
        #               "content": f"""<image>Given an X-ray image and a radiologist's description '{temp_sentence}', output a suitable disease category from <cat>Pneumonia, Fracture, Consolidation, Enlarged Cardiomegaly, No Finding, Pleural Other, Cardiomegaly, Pneumothorax, Atelectasis, Support Devices, Edema, Pleural Effusion, Lung Lesion, LungOpacity</cat> tags and explicitly indicate the anatomical area(s) you inspected from<area>abdomen, cardiac silhouette, left apical zone, left hilar structures, left lung, mediastinum, right apical zone, right hilar structures, right lung, whole lung, spine, trachea</area> tags with reasoning process. Limit your choice of anatomical areas to the single most relevant one. Do NOT list multiple sub-areas if a broader area like 'whole lung' adequately covers them. Finally, mimic a radiologist's interpretation pattern to identify the whole observation area. Your answer should strictly follow the format below: <think> Your reasoning process here... </think> <cat>Your chosen disease category here</cat> <area>Your chosen anatomical area(s) here (maximum two)</area><observation>[(xmin, ymin)(xmax, ymax)(centroidx, centroidy)]</observation>""",
        #               "role": "user"
        #             },
        #             {
        #               "content": f"{matching_row}<observation>{final_observation_area}</observation>",
        #               "role": "assistant"
        #             }
        #           ],
        #           "images": [
        #             f"./image_jpg/{row['id']}.jpg"
        #           ],
        #         })
#post filter is required if you have specific train test split
genseq = gen_seq()
result.extend(genseq)
gen_seq_localized = gen_seq_localized()
result.extend(gen_seq_localized)
import json
shuffled_data = shuffle(result)
print(len(shuffled_data))
with open('eye_tracking_training_dataset.json', 'w') as f:
    json.dump(shuffled_data, f)
