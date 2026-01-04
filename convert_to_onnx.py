import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType

# ==========================================
# 1. 配置 (必須與訓練時完全一致)
# ==========================================
INPUT_SIZE = 226        # 30 frames x (Pose + Left Hand + Right Hand)
NUM_CLASSES = 72       
MODEL_PATH = 'cnn_model.pth'
ONNX_OUTPUT_PATH = 'msl_model.onnx'
QUANTIZED_OUTPUT_PATH = 'msl_model_quant.onnx'

# ==========================================
# 2. 定義模型架構 (CNN) (必須與訓練代碼一致)
# ==========================================
class SignLanguageCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignLanguageCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Layer 3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Pooling & Classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x shape: (Batch, Sequence_Length, Features) -> (Batch, 30, 226)
        # CNN expects: (Batch, Channels, Length) -> (Batch, 226, 30)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)       # (Batch, 256, 1)
        x = x.flatten(1)       # (Batch, 256)
        x = self.dropout(x)
        
        x = self.fc(x)
        return x

# ==========================================
# 3. 執行轉換
# ==========================================
def convert():
    print(f"正在載入模型: {MODEL_PATH} ...")
    
    # 初始化模型 (CNN)
    model = SignLanguageCNN(INPUT_SIZE, NUM_CLASSES)
    
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

    # 導出模型 (使用 opset 13 並允許序列長度動態)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'seq_len'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"🎉 轉換完成！檔案已儲存為: {ONNX_OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ ONNX 導出失敗: {e}")
        return

    # ==========================================
    # 4. 量化模型 (加速推理)
    # ==========================================
    try:
        print(f"⚡ 正在量化模型 -> {QUANTIZED_OUTPUT_PATH} ...")
        # 避免對 Conv 層進行量化（某些 onnxruntime 版本可能不支援 ConvInteger），
        # 只對線性/MatMul/Gemm 類型的節點進行量化以保證可載入性
        quantize_dynamic(
            ONNX_OUTPUT_PATH,
            QUANTIZED_OUTPUT_PATH,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=['MatMul', 'Gemm']
        )
        print(f"✅ 量化完成！檔案已儲存為: {QUANTIZED_OUTPUT_PATH}")
        print("ℹ️ 注意：如果你想要對 Conv 也進行量化，請更新 onnxruntime 到最新版本，或使用靜態量化流程 (需校準資料)。")
    except Exception as e:
        print(f"⚠️ 量化失敗: {e}")
        print("🔧 嘗試：升級 onnxruntime (`pip install --upgrade onnxruntime`) 或使用只量化線性層的方法。")

    print("你可以使用 https://netron.app/ 查看生成的模型結構。")

if __name__ == "__main__":
    convert()