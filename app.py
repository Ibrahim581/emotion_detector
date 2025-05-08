import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import io
import numpy as np
from pydub import AudioSegment

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
model.load_state_dict(torch.load('emotion_cnn.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Emotion labels
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Preprocessing function
def preprocess_audio(uploaded_file):
    try:
        # Read file and convert to wav using pydub
        audio_segment = AudioSegment.from_file(uploaded_file)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

        # Export to a buffer in WAV format
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)

        # Load audio with librosa
        audio, sr = librosa.load(wav_io, sr=16000)
    except Exception as e:
        st.error(f"Audio loading failed: {str(e)}")
        st.stop()

    # Trim audio (remove leading/trailing silence)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Pad or trim to 3 seconds
    target_length = 3 * sr
    if len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    else:
        audio = audio[:target_length]

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)

    return mfccs.T

# Streamlit UI
st.title('Emotion Detection from Audio')
st.write('Upload an audio file and the model will predict the emotion.')

# File uploader
uploaded_file = st.file_uploader('Choose an audio file', type=['wav', 'mp3', 'ogg', 'm4a', 'flac', 'aac'])

if uploaded_file is not None:
    # Preprocess the audio file
    audio_data = preprocess_audio(uploaded_file)

    # Convert to tensor and add batch dimension
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

    # Check for minimum length
    if audio_tensor.shape[1] < 20:
        st.warning("Audio too short. Please upload at least 1-2 seconds of clear speech.")
    else:
        # Get prediction
        with torch.no_grad():
            output = model(audio_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        # Show result
        emotion = emotion_labels[predicted_class]
        st.write(f'**Predicted Emotion:** {emotion} ({confidence:.2%} confidence)')
