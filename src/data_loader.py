"""
Data Loading, Windowing, and Video Sequence Generator for GDI Prediction
Stanford CS 229: Automated Identification of Gait Abnormalities
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def load_image(filepath):
    """Load an image file into a NumPy array (H, W, C)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    img = Image.open(filepath)
    return np.array(img)

def windows(data_length, size=10, sample_stride=5):
    """
    Generator yielding (start, end) index tuples for sliding time windows.
    Matches the authors' windows() implementation.
    """
    start = 0
    while start + size <= data_length:
        yield int(start), int(start + size)
        start += sample_stride

def make_video_examples(df, window_size=10, sample_stride=5):
    """
    Groups consecutive frame paths into 10-frame sequences per training example,
    ensuring each sequence contains frames exclusively from the same patient.
    Matches the authors' hollywood() function.
    """
    sequences = []
    labels = []
    
    # Sort dataframe to guarantee chronological ordering within patient
    if 'frame_idx' in df.columns:
        df = df.sort_values(by=['patient_id', 'frame_idx']).reset_index(drop=True)
        
    for start, end in windows(len(df), window_size, sample_stride):
        # Verify all frames in window belong to the same patient
        start_patient = df['patient_id'].iloc[start]
        end_patient = df['patient_id'].iloc[end - 1]
        
        if start_patient == end_patient:
            file_paths = df['image_path'].iloc[start:end].values
            # Verify files exist
            if all(os.path.exists(p) for p in file_paths):
                frames = np.array([load_image(p) for p in file_paths])
                sequences.append(frames)
                labels.append(df['labels'].iloc[start])
                
    if len(sequences) == 0:
        return np.empty((0, window_size, 480, 640, 3)), np.empty((0,))
        
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

def extract_and_prepare_video_clip(video_path, max_frames=50, target_size=(640, 480)):
    """
    Reads an MP4 gait video, resizes frames to 640x480, formats into 
    DensePose IUV-like representations, and organizes into 10-frame sequences.
    """
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
            
        resized = cv2.resize(frame, target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # DensePose IUV synthesis
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        I = np.zeros_like(gray, dtype=np.uint8)
        I[mask > 0] = (gray[mask > 0] % 24) + 1
        U = np.zeros_like(gray, dtype=np.uint8)
        U[mask > 0] = gray[mask > 0]
        V = np.zeros_like(gray, dtype=np.uint8)
        V[mask > 0] = 255 - gray[mask > 0]
        
        iuv = cv2.merge([I, U, V])
        frames_list.append(iuv)
        count += 1
        
    cap.release()
    
    if len(frames_list) < 10:
        raise ValueError(f"Video {video_path} yielded fewer than 10 frames ({len(frames_list)} frames).")
        
    # Group into 10-frame clips with stride 5
    clips = []
    for s, e in windows(len(frames_list), size=10, sample_stride=5):
        clips.append(np.array(frames_list[s:e], dtype=np.float32))
        
    return np.array(clips, dtype=np.float32)

def interpret_gdi_score(gdi):
    """
    Clinical interpretation of the Gait Deviation Index (GDI).
    According to Schwartz & Rozumalski (2008):
    - GDI >= 100: Normal / typical walking gait
    - Each 10-point reduction corresponds to 1 standard deviation away from normal
    - 90 - 100: Mild gait deviation (within 1 SD)
    - 80 - 90: Moderate gait deviation (1-2 SD)
    - 70 - 80: Significant gait deviation (2-3 SD)
    - < 70: Severe gait pathology (> 3 SD)
    """
    if gdi >= 100:
        return "Normal / Typical Gait (GDI >= 100)"
    elif gdi >= 90:
        return "Mild Gait Deviation (1 SD from normal)"
    elif gdi >= 80:
        return "Moderate Gait Deviation (1 - 2 SD from normal)"
    elif gdi >= 70:
        return "Significant Gait Deviation (2 - 3 SD from normal)"
    else:
        return "Severe Gait Pathology (> 3 SD from normal)"
