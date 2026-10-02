import os
import json
import cv2
import pandas as pd
import random
from tqdm import tqdm

# Paths
labels_dir = '/kaggle/input/datasets/nahidsiddique/something-something-v2/20bn-something-something-download-package-labels/labels'
videos_dir = '/kaggle/input/datasets/nahidsiddique/something-something-v2/20bn-something-something-v2/20bn-something-something-v2'
output_frames_dir = '/kaggle/working/frames'
output_csv_dir = '/kaggle/working/'

os.makedirs(output_frames_dir, exist_ok=True)

# 1. Define our target classes
TARGET_CLASSES = [
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Moving [something] down",
    "Tearing [something] into two pieces"
]

# Map template to an integer class ID for our 5 classes
class_to_id = {cls: i for i, cls in enumerate(TARGET_CLASSES)}

SAMPLES_PER_CLASS_TRAIN = 1000  # Adjust if you want more/less data
SAMPLES_PER_CLASS_VAL = 100
NUM_FRAMES = 8 # Number of frames to extract per video

def get_subset(json_path, samples_per_class):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Group by class
    class_groups = {cls: [] for cls in TARGET_CLASSES}
    for item in data:
        template = item.get('template')
        if template in TARGET_CLASSES:
            class_groups[template].append(item['id'])
            
    # Sample
    subset_records = []
    for cls, video_ids in class_groups.items():
        sampled_ids = random.sample(video_ids, min(len(video_ids), samples_per_class))
        for vid_id in sampled_ids:
            subset_records.append({'video_id': vid_id, 'class_name': cls, 'class_id': class_to_id[cls]})
            
    return subset_records

def extract_frames(video_records, split_name):
    print(f"Extracting frames for {split_name}...")
    valid_records = []
    
    for record in tqdm(video_records):
        vid_id = record['video_id']
        video_path = os.path.join(videos_dir, f"{vid_id}.webm")
        
        if not os.path.exists(video_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < NUM_FRAMES:
            cap.release()
            continue
            
        # Calculate indices for evenly spaced frames
        indices = [int(i * (total_frames - 1) / (NUM_FRAMES - 1)) for i in range(NUM_FRAMES)]
        
        vid_out_dir = os.path.join(output_frames_dir, str(vid_id))
        os.makedirs(vid_out_dir, exist_ok=True)
        
        frames_saved = 0
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in indices:
                # If an index is repeated due to rounding, save multiple to match NUM_FRAMES
                counts = indices.count(current_frame)
                for _ in range(counts):
                    frame_path = os.path.join(vid_out_dir, f"frame_{frames_saved:02d}.jpg")
                    # Resize to 224x224 (standard for DINO/CLIP) to save disk space
                    frame_resized = cv2.resize(frame, (224, 224))
                    cv2.imwrite(frame_path, frame_resized)
                    frames_saved += 1
            current_frame += 1
            
        cap.release()
        
        if frames_saved == NUM_FRAMES:
            valid_records.append(record)
            
    return valid_records

# 2. Sample data
random.seed(42)
train_records = get_subset(os.path.join(labels_dir, 'train.json'), SAMPLES_PER_CLASS_TRAIN)
val_records = get_subset(os.path.join(labels_dir, 'validation.json'), SAMPLES_PER_CLASS_VAL)

# 3. Extract frames
valid_train = extract_frames(train_records, 'train')
valid_val = extract_frames(val_records, 'val')

# 4. Save to CSV
train_df = pd.DataFrame(valid_train)
val_df = pd.DataFrame(valid_val)

train_df.to_csv(os.path.join(output_csv_dir, 'train_subset.csv'), index=False)
val_df.to_csv(os.path.join(output_csv_dir, 'val_subset.csv'), index=False)

print(f"Extraction complete! Saved {len(train_df)} train videos and {len(val_df)} val videos.")
print("CSV files saved to /kaggle/working/train_subset.csv and val_subset.csv")