import torch
import torchvision.models as models
import numpy as np
import cv2

# Define the list of C3D/I3D models
k = 3  # Number of videos and CNNs
#cnns = [models.resnet18(pretrained=True) for _ in range(k)]

cnnn = [models.C3D(pretrained=True) for _ in range(k)] # Feature Extractor

# Set the models to evaluation mode
for cnn in cnns:
    cnn.eval()

# Function to preprocess video frames
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = cv2.resize(frame, (224, 224))  # Resize to match input size of the CNN model
    frame = frame.transpose((2, 0, 1))  # Transpose dimensions for PyTorch (C, H, W)
    frame = frame / 255.0  # Normalize pixel values between 0 and 1
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = torch.from_numpy(frame).float()  # Convert to torch tensor
    return frame

# Function to extract features from video segments
def extract_segment_features(video_path, segment_length=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    num_segments = int(total_frames / segment_length)
    features = []
    for segment in range(num_segments):
        segment_frames = []
        for _ in range(segment_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = preprocess_frame(frame)
            segment_frames.append(frame)
        segment_frames = torch.cat(segment_frames)
        with torch.no_grad():
            segment_features = cnns[i](segment_frames)
        segment_features = segment_features * rank_scores[i]
        features.append(segment_features)
    cap.release()
    return torch.cat(features)

# List to store the extracted features
features = []

# List of rank scores for each video
rank_scores = [0.2, 0.5, 0.3]  # Replace with the rank scores of your videos

# Load and extract features from the videos
video_paths = ['accident_33.mp4', 'accident_32.mp4', 'accident_31.mp4']

for i, video_path in enumerate(video_paths):
    # Extract segment features for current video
    video_features = extract_segment_features(video_path, segment_length=10)

    # Append the segment features to the list
    features.append(video_features)

# Perform average pooling over the features
average_features = torch.mean(torch.stack(features), dim=0)

# Convert the average features to a NumPy array
average_features_np = average_features.numpy()

# Save the average features as a single .npy file
np.save('average_features_2.npy', average_features_np)