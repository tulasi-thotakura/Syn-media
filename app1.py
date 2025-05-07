import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import zipfile
import cv2
import tempfile
import librosa
from tensorflow.keras.models import load_model
import tensorflow as tf
from torchvision import models, transforms
from torch import nn
from torch.utils.data import Dataset
import face_recognition
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---- Load Models ----
@st.cache_resource
def load_all_models():
    vid_model = load_model("forgery_model_me.hdf5")
    aud_model = load_model("model-resnet18.pth")
    return vid_model, aud_model

vid_model, aud_model = load_all_models()

# ---- Page Navigation ----
st.sidebar.title("Synthetic Media Detection")
page = st.sidebar.selectbox("Choose Type", ["Detect Image", "Detect Video", "Detect Audio", "Detect Video (Best)"])

# ---- Image Detection Page ----
if page == "Detect Image":
    st.title("DeepFake Image Detection - Real vs Fake")
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).eval().to(DEVICE)
    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1).to(DEVICE)
    checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert('RGB')
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        face = mtcnn(input_image)
        if face is None:
            st.error('No face detected')
        else:
            face = face.unsqueeze(0)
            face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
            face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
            face = face.to(DEVICE).float() / 255.0
            with torch.no_grad():
                output = torch.sigmoid(model(face).squeeze(0))
                prediction = "Real" if output.item() < 0.5 else "Fake"
                confidence = output.item() if prediction == "Fake" else 1 - output.item()
            st.image(face_image_to_plot, caption="Detected Face", use_column_width=False)
            st.write(f"Prediction: **{prediction}**")
            st.write(f"Confidence: {confidence:.2f}")

# ---- Video Detection Page ----
elif page == "Detect Video":
    st.title("Video Forgery Detection")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    def detect_video_forgery(vid_src):
        vid = []
        sumFrames = 0
        cap = cv2.VideoCapture(vid_src)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_CUBIC)
            sumFrames += 1
            vid.append(resized_frame)
        cap.release()

        Xtest = np.array(vid)
        output = vid_model.predict(Xtest)
        output = output.reshape((-1))
        results = [1 if i > 0.5 else 0 for i in output]
        no_of_forged = sum(results)
        if no_of_forged <= 0:
            return {'result': 'Authentic', 'f_frames': 0}
        else:
            return {'result': 'Forged', 'f_frames': no_of_forged}

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_vid:
            temp_vid.write(uploaded_file.read())
            temp_path = temp_vid.name
        st.video(uploaded_file)
        if st.button("Check for Forgery"):
            with st.spinner("Processing..."):
                result = detect_video_forgery(temp_path)
                if result['result'] == 'Authentic':
                    st.success("The video is Authentic. No forged frames found.")
                else:
                    st.error(f"The video is Forged. Forged frames: {result['f_frames']}")

# ---- Audio Detection Page ----
elif page == "Detect Audio":
    st.title("Fake Audio Detection")
    max_duration = 5
    max_length = 216

    def pad_mfcc(mfcc, max_len):
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc

    def detect_fake(filename):
        sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast", duration=max_duration)
        mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
        mfcc_features = pad_mfcc(mfcc_features, max_length)
        mfcc_features = np.expand_dims(mfcc_features, axis=-1)
        mfcc_features = np.expand_dims(mfcc_features, axis=0)
        result_array = aud_model.predict(mfcc_features)
        result_classes = ["FAKE", "REAL"]
        result = np.argmax(result_array[0])
        confidence = result_array[0][result]
        return result_classes[result], confidence

    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/mp3")
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.read())
        label, confidence = detect_fake("temp_audio.mp3")
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")

# ---- Best Video Detection Page (from First Code) ----
elif page == "Detect Video (Best)":
    st.title("Deepfake Video Detection (Best)")
    
    class Model(nn.Module):
        def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
            super(Model, self).__init__()
            model = models.resnext50_32x4d(pretrained=True)
            self.model = nn.Sequential(*list(model.children())[:-2])
            self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
            self.relu = nn.LeakyReLU()
            self.dp = nn.Dropout(0.4)
            self.linear1 = nn.Linear(2048, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            batch_size, seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w)
            fmap = self.model(x)
            x = self.avgpool(fmap)
            x = x.view(batch_size, seq_length, 2048)
            x_lstm, _ = self.lstm(x, None)
            return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

    class validation_dataset(Dataset):
        def __init__(self, video_names, sequence_length=60, transform=None):
            self.video_names = video_names
            self.transform = transform
            self.count = sequence_length

        def __len__(self):
            return len(self.video_names)

        def __getitem__(self, idx):
            video_path = self.video_names[idx]
            frames = []
            a = int(100 / self.count)
            first_frame = np.random.randint(0, a)
            for i, frame in enumerate(self.frame_extract(video_path)):
                faces = face_recognition.face_locations(frame)
                try:
                    top, right, bottom, left = faces[0]
                    frame = frame[top:bottom, left:right, :]
                except:
                    pass
                frames.append(self.transform(frame))
                if len(frames) == self.count:
                    break
            frames = torch.stack(frames)
            frames = frames[:self.count]
            return frames.unsqueeze(0)

        def frame_extract(self, path):
            vidObj = cv2.VideoCapture(path)
            success = 1
            while success:
                success, image = vidObj.read()
                if success:
                    yield image

    def predict(model, img):
        sm = nn.Softmax()
        fmap, logits = model(img)
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        return [int(prediction.item()), confidence]

    def detectFakeVideo(videoPath):
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        video_dataset = validation_dataset([videoPath], sequence_length=20, transform=transform)
        model = Model(2)
        model.load_state_dict(torch.load("df_model.pt", map_location=torch.device('cpu')))
        model.eval()
        prediction = predict(model, video_dataset[0])
        return prediction

    uploaded_file = st.file_uploader("Upload a video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        st.video(video_path)
        with st.spinner("Analyzing..."):
            result = detectFakeVideo(video_path)
            label = "REAL" if result[0] == 1 else "FAKE"
            confidence = result[1]
            st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")
        os.remove(video_path)
