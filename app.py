import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from collections import Counter

# ==========================================
# 1. 配置與模型 (保持不變)
# ==========================================
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

# 載入模型
model = CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)
try:
    # 雲端通常是 CPU，所以強制 map_location='cpu'
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Model Loaded!")
except Exception as e:
    print(f"Model Load Error: {e}")

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 2. 核心處理邏輯
# ==========================================
def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[:25]]).flatten()
    else:
        pose = np.zeros(25*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def predict_frame(image, state):
    # state 是一個字典，用來在不同幀之間傳遞數據
    if state is None:
        state = {"sequence": [], "predictions": [], "sentence": "Waiting...", "frame_count": 0}
    
    if image is None:
        return None, state

    # 1. 影像前處理
    # Gradio 傳入的是 RGB，MediaPipe 也吃 RGB
    image.flags.writeable = False
    
    # 初始化 Holistic (每次都重新初始化會慢，但在函數式編程中比較安全)
    # 為了效能，Hugging Face 會自動優化
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image)
    
    image.flags.writeable = True
    
    # 畫圖
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # 減少繪製身體以提升速度
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    state["frame_count"] += 1
    
    # 2. 預測邏輯 (每 2 幀處理一次)
    if state["frame_count"] % 2 == 0:
        keypoints = extract_keypoints(results)
        state["sequence"].append(keypoints)
        state["sequence"] = state["sequence"][-30:] # 保持 30 幀

        if len(state["sequence"]) == 30:
            input_tensor = torch.tensor(np.expand_dims(state["sequence"], axis=0), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                res = model(input_tensor)
            
            probs = torch.softmax(res, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            max_prob = probs[pred_idx]
            
            state["predictions"].append(pred_idx)
            state["predictions"] = state["predictions"][-10:]
            
            # 穩定化邏輯
            if len(state["predictions"]) > 0:
                most_common_id, frequency = Counter(state["predictions"]).most_common(1)[0]
                if frequency >= 8 and max_prob > 0.7:
                    state["sentence"] = gestures[most_common_id]
                    # 綠色條條
                    cv2.rectangle(image, (0,0), (int(max_prob*200), 40), (0,255,0), -1)
                elif max_prob > 0.7:
                    # 黃色條條
                    cv2.rectangle(image, (0,0), (int(max_prob*200), 40), (0,255,255), -1)

    # 3. 繪製文字
    cv2.rectangle(image, (0, 40), (640, 80), (245, 117, 16), -1)
    cv2.putText(image, state["sentence"], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 鏡像翻轉回傳
    return cv2.flip(image, 1), state

# ==========================================
# 3. Gradio 介面
# ==========================================
with gr.Blocks(title="MSL Recognition AI") as demo:
    gr.Markdown("# 🇲🇾 Malaysian Sign Language Recognition")
    gr.Markdown("Stand back and show your upper body. Perform signs slowly.")
    
    with gr.Row():
        with gr.Column():
            # sources=["webcam"] 開啟攝像頭
            # streaming=True 開啟即時流模式
            input_video = gr.Image(sources=["webcam"], streaming=True, label="Input Camera")
        with gr.Column():
            output_video = gr.Image(label="AI Output")
    
    # 用來記憶狀態的變數
    state = gr.State()
    
    # 當輸入影像改變時，呼叫 predict_frame
    input_video.stream(
        predict_frame, 
        [input_video, state], 
        [output_video, state]
    )

if __name__ == "__main__":
    demo.launch()