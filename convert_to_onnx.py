import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType

# ==========================================
# 1. 配置 (必須與訓練時完全一致)
# ==========================================
INPUT_SIZE = 226        # 30 frames x (Pose + Left Hand + Right Hand)
HIDDEN_SIZE = 64
NUM_CLASSES = 90        # ⚠️ 注意：如果你只訓練了 10 個詞，這裡要改成 10
MODEL_PATH = 'baseline_model.pth'
ONNX_OUTPUT_PATH = 'msl_model.onnx'
QUANTIZED_OUTPUT_PATH = 'msl_model_quant.onnx'

# ==========================================
# 2. 定義模型架構 (必須與訓練代碼完全一致)
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
        # LSTM 層
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        # 取最後一個時間點的特徵 (Last Timestep)
        x = x[:, -1, :] 
        
        # 全連接層
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x

# ==========================================
# 3. 執行轉換
# ==========================================
def convert():
    print(f"正在載入模型: {MODEL_PATH} ...")
    
    # 初始化模型
    model = CustomLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    
    # 載入權重 (強制使用 CPU 載入，避免 CUDA 報錯)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # 設定為評估模式，這對導出 ONNX 很重要
        print("✅ 模型載入成功！")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        print("請檢查 NUM_CLASSES 是否正確，或 .pth 檔案路徑是否正確。")
        return

    # 建立一個虛擬輸入 (Dummy Input)
    # 形狀: (Batch_Size, Sequence_Length, Features) -> (1, 30, 226)
    # PyTorch 需要跑一次數據才能知道模型的結構
    dummy_input = torch.randn(1, 30, INPUT_SIZE)

    print(f"正在轉換為 ONNX: {ONNX_OUTPUT_PATH} ...")
    
    # 導出模型
    torch.onnx.export(
        model,                      # 你的模型
        dummy_input,                # 虛擬輸入
        ONNX_OUTPUT_PATH,           # 輸出檔名
        export_params=True,         # 是否儲存權重
        opset_version=12,           # ONNX 版本 (11 或 12 比較穩定)
        do_constant_folding=True,   # 優化常數折疊
        input_names=['input'],      # 輸入層名稱
        output_names=['output'],    # 輸出層名稱
        dynamic_axes={              # 設定動態維度 (讓 Batch Size 可以變動)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"🎉 轉換完成！檔案已儲存為: {ONNX_OUTPUT_PATH}")

    # ==========================================
    # 4. 量化模型 (加速推理)
    # ==========================================
    try:
        print(f"⚡ 正在量化模型 -> {QUANTIZED_OUTPUT_PATH} ...")
        quantize_dynamic(ONNX_OUTPUT_PATH, QUANTIZED_OUTPUT_PATH, weight_type=QuantType.QInt8)
        print(f"✅ 量化完成！檔案已儲存為: {QUANTIZED_OUTPUT_PATH}")
    except Exception as e:
        print(f"⚠️ 量化失敗: {e}")

    print("你可以使用 https://netron.app/ 查看生成的模型結構。")

if __name__ == "__main__":
    convert()