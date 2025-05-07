import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import io
import numpy as np

# Model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(256, 128)  # 128 mean + 128 max
        self.fc2 = nn.Linear(128, 8)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, 40) → (B, 40, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x_mean = x.mean(dim=2)
        x_max = x.max(dim=2).values
        x = torch.cat((x_mean, x_max), dim=1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# Load the model
model = EmotionCNN()
model.load_state_dict(torch.load('emotion_cnn.pth', map_location = torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Emotion mapping
emotion_mapping = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

# Preprocessing function
def preprocess_audio(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=16000)

    # Trim audio (remove leading/trailing silence)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Pad to 3 seconds (if shorter)
    if len(audio) < 3 * sr:
        padding = 3 * sr - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    # Resample to 16kHz (if needed)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)

    # Optionally extract Mel spectrogram
    # melspectrogram = librosa.feature.melspectrogram(y=audio, sr=16000)

    # Return the MFCC features (or mel spectrogram, based on your preference)
    return mfccs.T  # Transpose so that shape is [time_steps, n_mfcc] (better for models)

# Streamlit UI
st.title('Emotion Detection from Audio')
st.write('Upload an audio file and the model will predict the emotion.')

# File uploader
audio_file = st.file_uploader('Choose an audio file', type=['wav'])

if audio_file is not None:
    # Preprocess the audio file
    audio_data = preprocess_audio(audio_file)
    
    # Convert to tensor and add batch dimension
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        output = model(audio_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Map the predicted class to the emotion
    emotion = list(emotion_mapping.keys())[list(emotion_mapping.values()).index(predicted_class)]
    
    st.write(f'Predicted Emotion: {emotion}')
