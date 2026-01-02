import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from collections import Counter
import av

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# 必須與訓練時的順序完全一致
gestures = np.array([
    'abang', 'ada', 'ambil', 'anak_lelaki', 'anak_perempuan', 'apa', 'apa_khabar', 'arah', 
    'assalamualaikum', 'ayah', 'bagaimana', 'bahasa_isyarat', 'baik', 'baik_2', 'baca', 
    'bapa', 'bapa_saudara', 'bas', 'bawa', 'beli', 'beli_2', 'berapa', 'berjalan', 'berlari', 
    'bila', 'bola', 'boleh', 'bomba', 'buang', 'buat', 'curi', 'dapat', 'dari', 'emak', 
    'emak_saudara', 'hari', 'hi', 'hujan', 'jahat', 'jam', 'jangan', 'jumpa', 'kacau', 
    'kakak', 'keluarga', 'kereta', 'kesakitan', 'lelaki', 'lemak', 'lupa', 'main', 'makan', 
    'mana', 'marah', 'mari', 'masa', 'masalah', 'minum', 'mohon', 'nasi', 'nasi_lemak', 
    'panas', 'panas_2', 'pandai', 'pandai_2', 'payung', 'pen', 'pensil', 'perempuan', 
    'pergi', 'pergi_2', 'perlahan', 'perlahan_2', 'pinjam', 'polis', 'pukul', 'ribut', 
    'sampai', 'saudara', 'sejuk', 'sekolah', 'siapa', 'sudah', 'suka', 'tandas', 'tanya', 
    'teh_tarik', 'teksi', 'tidur', 'tolong'
])
gestures = np.sort(gestures)

MODEL_PATH = 'baseline_model.pth' 
INPUT_SIZE = 226
HIDDEN_SIZE = 64
NUM_CLASSES = len(gestures)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. DEFINE MODEL (Must match training)
# ==========================================
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :] 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x

# 快取模型以提升效能
@st.cache_resource
def load_model():
    model = CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)
    try:
        # map_location='cpu' 確保在沒有 GPU 的雲端也能跑
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        return None

# ==========================================
# 3. HELPER FUNCTION
# ==========================================
def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[:25]]).flatten()
    else:
        pose = np.zeros(25*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# ==========================================
# 4. WEBRTC PROCESSOR
# ==========================================
class SignLanguageProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        self.sequence = []
        self.predictions = []
        self.sentence = "Waiting..."
        self.threshold = 0.7
        self.frame_counter = 0
        self.SKIP_FRAMES = 2 # Process every 2nd frame to save CPU on cloud

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 鏡像翻轉，讓使用者體驗更自然
        img = cv2.flip(img, 1)
        
        # MediaPipe Detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.holistic.process(img_rgb)
        img_rgb.flags.writeable = True

        # Draw Landmarks
        self.mp_drawing.draw_landmarks(img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        # 減少繪製身體連線以保持畫面乾淨，也可以根據需求加回來
        self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

        self.frame_counter += 1

        # Prediction Logic
        if self.model and self.frame_counter % self.SKIP_FRAMES == 0:
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:] # Keep last 30 frames

            if len(self.sequence) == 30:
                input_tensor = torch.tensor(np.expand_dims(self.sequence, axis=0), dtype=torch.float32).to(DEVICE)
                
                with torch.no_grad():
                    res = self.model(input_tensor)
                
                probs = torch.softmax(res, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                max_prob = probs[pred_idx]

                # Stabilization Logic
                self.predictions.append(pred_idx)
                self.predictions = self.predictions[-10:]

                if len(self.predictions) > 0:
                    most_common_id, frequency = Counter(self.predictions).most_common(1)[0]
                    # 如果同一個結果在過去10幀出現超過8次，才更新文字
                    if frequency >= 8 and max_prob > self.threshold:
                        self.sentence = gestures[most_common_id]
                        cv2.rectangle(img, (0,0), (int(max_prob*200), 40), (0,255,0), -1)
                    elif max_prob > self.threshold:
                        cv2.rectangle(img, (0,0), (int(max_prob*200), 40), (0,255,255), -1)

        # UI Overlay
        cv2.rectangle(img, (0, 40), (640, 80), (245, 117, 16), -1)
        cv2.putText(img, self.sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="MSL Recognition", layout="wide")

st.title("🇲🇾 Malaysian Sign Language Recognition")
st.markdown("### AI-Powered Real-time Translator")

# 檢查模型是否載入成功
if load_model() is None:
    st.error("⚠️ Error: 'baseline_model.pth' not found! Please check file structure.")

col1, col2 = st.columns([2, 1])

with col1:
    # Google STUN Server 設定 (解決手機/防火牆問題)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="msl-translator",
        video_processor_factory=SignLanguageProcessor,
        mode="sendrecv",
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("#### User Guide")
    st.info("""
    1. Click **START** and allow camera access.
    2. Step back to show your **upper body**.
    3. Perform gestures slowly and clearly.
    4. Ensure good lighting.
    """)
    
    st.markdown("#### Supported Glosses")
    st.text_area("List", ", ".join(gestures), height=300)