# Streamlit app using V-JEPA for feature extraction + overall best classifier

import streamlit as st
import torch
from transformers import AutoModel, AutoVideoProcessor
import torchvision.io as io
import os
import tempfile
import subprocess
import joblib # 導入 joblib
import numpy as np

# --- Streamlit 頁面配置 ---
st.set_page_config(page_title="V-JEPA Video Classifier", layout="centered")

st.title("🎬 V-JEPA 2 + 影片分類器")
st.write("上傳影片，自動轉成 MP4，抽取特徵並用**訓練後整體最佳模型**進行分類。")

# --- 載入 label_map.txt 的函數 ---
def load_label_map(path="label_map.txt"):
    mapping = {}
    try:
        with open(path, "r") as f:
            for line in f:
                idx, label = line.strip().split()
                mapping[int(idx)] = label
    except FileNotFoundError:
        st.error("❌ 找不到 `label_map.txt`，請先執行訓練腳本以生成該檔案。")
        return None
    return mapping

label_map = load_label_map()
if label_map is None:
    st.stop() # 如果沒有 label_map，停止應用程式

# --- 載入 V-JEPA 和分類器模型 ---
@st.cache_resource # 使用 Streamlit 的快取，避免每次執行都重新載入大模型
def load_models():
    hf_repo = "facebook/vjepa2-vitg-fpc64-256"
    
    st.write("⏳ 載入 V-JEPA 特徵提取模型中...")
    processor = AutoVideoProcessor.from_pretrained(hf_repo)
    base_model = AutoModel.from_pretrained(hf_repo)
    base_model.cuda().eval() # 將模型移到 GPU 並設為評估模式

    st.write("⏳ 載入訓練後**整體最佳**分類器模型 (`vjepa_classifier_overall_best.pt`) 中...")
    try:
        # **這裡就是關鍵的修改！** 載入整體最佳模型
        classifier_model = joblib.load("vjepa_classifier_overall_best.pt")
        st.success("✅ 模型載入完成！")
    except FileNotFoundError:
        st.error("❌ 找不到 `vjepa_classifier_overall_best.pt` 模型檔案。請確認您已運行訓練腳本並生成該檔案。")
        return None, None, None
    except Exception as e:
        st.error(f"❌ 載入分類器模型時發生錯誤: {e}")
        return None, None, None
        
    return processor, base_model, classifier_model

processor, base_model, classifier_model = load_models()

# 如果模型載入失敗，則停止應用程式
if classifier_model is None:
    st.stop()

# --- 視訊轉 MP4 函數 ---
# Streamlit 在某些環境下對於非 MP4 格式支援不好，轉檔是個好習慣
def convert_to_mp4(input_path):
    output_path = tempfile.mktemp(suffix=".mp4")
    # 使用 ffmpeg 進行轉碼，確保輸出影片的編碼和格式兼容性
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path
    ]
    try:
        # 使用 st.empty() 和 progress 條顯示轉檔進度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 簡易進度條 (可能不精確，但提供使用者反饋)
        total_duration = 10 # 假設最長轉檔時間為 10 秒，用於進度條粗略估計
        for i in range(1, 101):
            progress_bar.progress(i)
            status_text.text(f"🎥 影片轉檔中... {i}%")
            # 實際應用中，可以解析 ffmpeg 的 stderr 輸出以獲取更精確的進度
            # 但這會使代碼更複雜，這裡只為 Streamlit 演示一個簡單的進度條
            if i % 20 == 0:
                torch.cuda.empty_cache() # 嘗試釋放 GPU 記憶體

        stdout, stderr = process.communicate(timeout=600) # 設定超時
        if process.returncode != 0:
            st.error(f"❌ 影片轉檔失敗 (ffmpeg 錯誤碼: {process.returncode}):\n{stderr.decode()}")
            return None
        st.success("✅ 影片轉檔完成！")
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"❌ 影片轉檔失敗: {e.stderr.decode()}")
        return None
    except subprocess.TimeoutExpired:
        process.kill()
        st.error("❌ 影片轉檔超時，請檢查影片檔案或嘗試較短的影片。")
        return None
    except Exception as e:
        st.error(f"❌ 影片轉檔時發生未知錯誤: {str(e)}")
        return None
    finally:
        # 清除進度條和狀態文本
        progress_bar.empty()
        status_text.empty()


# --- 抽取特徵函數 ---
@torch.no_grad() # 確保不計算梯度，節省記憶體
def extract_features(video_tensor):
    # 將影片幀移動到 CUDA 設備
    # video_tensor 的形狀可能是 (T, H, W, C)，需要轉換為處理器期望的格式 (例如列表 of numpy arrays)
    # 或者如果 processor 能直接處理 torch.Tensor，確保其在 CPU 或 GPU 上
    
    # 這裡我們假設 processor.preprocess 能夠處理 list of numpy arrays (H,W,C)
    # 因此需要將 torch.Tensor (T, H, W, C) 轉換為 list of numpy arrays
    clip_list = [frame.cpu().numpy() for frame in video_tensor]
    
    inputs = processor(clip_list, return_tensors="pt")
    # 將處理後的輸入移動到 CUDA 設備
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    base_output = base_model(**inputs)
    features = base_output.last_hidden_state # (batch_size, sequence_length, hidden_size)
    
    # 對序列維度進行平均池化，得到固定大小的特徵向量 (batch_size, hidden_size)
    # Permute 從 (B, S, H) 到 (B, H, S) 以適應 AdaptiveAvgPool1d
    pooled = torch.nn.functional.adaptive_avg_pool1d(features.permute(0, 2, 1), 1).squeeze(-1)
    
    # 返回特徵，確保是 CPU 上的 numpy 陣列，且形狀適合 Scikit-learn 模型 (1, feature_dim)
    return pooled.cpu().numpy().reshape(1, -1) # 確保是 2D array for predict


# --- 推論流程 ---
def predict_video(video_path):
    frames_per_clip = 16 # V-JEPA 模型通常會期望固定數量的幀
    
    st.write("🔍 讀取影片並準備特徵提取...")
    try:
        # io.read_video 返回 (num_frames, H, W, C), pts, meta
        video_tensor, _, _ = io.read_video(video_path, pts_unit='sec')
        
        # 確保影片幀數至少為 1
        if video_tensor.shape[0] == 0:
            st.error("❌ 影片沒有任何可讀取的幀，請檢查影片檔案是否損壞。")
            return "Error"

        # 如果影片幀數少於 `frames_per_clip`，則進行填充
        if video_tensor.shape[0] < frames_per_clip:
            st.warning(f"影片幀數 ({video_tensor.shape[0]}) 少於模型所需 ({frames_per_clip})，將進行幀填充。")
            pad_amount = frames_per_clip - video_tensor.shape[0]
            # 重複最後一幀進行填充
            clip = torch.cat([video_tensor, video_tensor[-1:].repeat(pad_amount, 1, 1, 1)])
        else:
            # 如果影片幀數過多，則從中間截取 `frames_per_clip` 幀
            start = (video_tensor.shape[0] - frames_per_clip) // 2
            clip = video_tensor[start : start + frames_per_clip]
        
        # 確保 clip 已經在 CPU 上，因為 extract_features 會處理到 GPU
        clip = clip.cpu()

        st.write("🚀 抽取影片特徵中...")
        feature = extract_features(clip) # 提取特徵

        st.write("🧠 進行分類預測...")
        # 分類模型通常期望 2D 數組 (n_samples, n_features)
        # feature 已經被 reshape 成 (1, feature_dim)
        pred_id = classifier_model.predict(feature)[0]
        
        predicted_label = label_map.get(pred_id, "未知類別")
        
        # 如果分類器支持 predict_proba，則顯示機率
        if hasattr(classifier_model, 'predict_proba'):
            probabilities = classifier_model.predict_proba(feature)[0]
            st.write("各類別預測機率:")
            prob_dict = {label_map[i]: f"{prob:.2%}" for i, prob in enumerate(probabilities)}
            st.json(prob_dict)

        return predicted_label

    except Exception as e:
        st.error(f"推論時發生錯誤: {str(e)}")
        # 更詳細的錯誤日誌可以在後端查看
        print(f"DEBUG: Error during prediction: {e}", exc_info=True)
        return "Error"

# --- Streamlit UI 邏輯 ---
uploaded_file = st.file_uploader("請上傳影片檔（avi, mp4, mov, mkv, flv）", type=["avi", "mp4", "mov", "mkv", "flv"])

if uploaded_file is not None:
    # 使用 tempfile 處理上傳檔案，確保跨平台兼容性
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        uploaded_video_path = tmp_file.name

    st.video(uploaded_video_path) # 顯示原始上傳影片

    # 確保是 MP4 格式，ffmpeg 會將影片轉碼為 MP4
    mp4_video_path = convert_to_mp4(uploaded_video_path)
    
    # 刪除臨時上傳的原始檔案
    os.unlink(uploaded_video_path)

    if mp4_video_path:
        with st.spinner("✨ 正在使用最佳模型進行預測..."):
            predicted_label = predict_video(mp4_video_path)
            
        if predicted_label != "Error":
            st.success(f"✅ 預測結果：這個影片的活動是 **{predicted_label}**！")
        
        # 預測完成後刪除臨時的 MP4 檔案
        if os.path.exists(mp4_video_path):
            os.unlink(mp4_video_path)