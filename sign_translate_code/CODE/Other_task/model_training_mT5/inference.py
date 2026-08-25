import pandas as pd
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import os
from tqdm import tqdm # 用於顯示進度條，更友善

# --- 1. 設定路徑 ---
# 模型和資料來源的資料夾路徑
MODEL_DIR = "20250618_0705_mt5_prompt_all/best_model"
# 要進行推論的輸入檔案
INPUT_CSV_PATH = os.path.join(MODEL_DIR, "best_train.csv")
# 儲存預測結果的輸出檔案
OUTPUT_CSV_PATH = os.path.join(MODEL_DIR, "train_set_predictions.csv")

def run_inference():
    """
    載入已訓練好的模型，對指定的 CSV 檔案進行推論，並儲存結果。
    """
    # --- 2. 檢查路徑是否存在 ---
    if not os.path.exists(MODEL_DIR):
        print(f"錯誤：找不到模型資料夾 '{MODEL_DIR}'。")
        print("請確認此腳本與 '20250618_0705_mt5_prompt_all' 資料夾在同一個目錄中。")
        return

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"錯誤：在模型資料夾中找不到輸入檔案 '{INPUT_CSV_PATH}'。")
        return

    # --- 3. 載入模型和 Tokenizer ---
    print(f"正在從 '{MODEL_DIR}' 載入模型和 Tokenizer...")
    
    # 設定裝置 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 從儲存的路徑載入 tokenizer 和模型
    # from_pretrained 會自動處理 special tokens 的問題
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval() # 將模型設定為評估模式

    # --- 4. 讀取資料 ---
    print(f"正在讀取資料: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    # 確保 `input_text` 和 `target_text` 欄位存在
    if "input_text" not in df.columns or "target_text" not in df.columns:
        print("錯誤：CSV 檔案中缺少 'input_text' 或 'target_text' 欄位。")
        return

    # --- 5. 執行推論 ---
    predictions = []
    references = []

    print(f"開始對 {len(df)} 筆資料進行推論...")
    
    # 使用 with torch.no_grad() 進行推論以節省記憶體並加速
    with torch.no_grad():
        # 使用 tqdm 顯示進度條
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Predictions"):
            input_text = row["input_text"]
            target_text = row["target_text"] # 也就是 reference
            
            # 使用 tokenizer 將輸入文字編碼
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512 # 與訓練時相同
            ).to(device)

            # 模型生成預測結果
            # max_length 建議設定為與訓練時 target 的長度相近
            generated_ids = model.generate(**inputs, max_length=128) 
            
            # 將生成的 token IDs 解碼回文字
            predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            predictions.append(predicted_text)
            references.append(target_text)

    # --- 6. 儲存結果 ---
    print("\n推論完成！")
    
    # 顯示前 5 筆預測結果以供檢視
    print("\n前 5 筆預測 vs. 參考答案:")
    for i in range(min(5, len(predictions))):
        print("-" * 20)
        print(f"  Input:      {df['input_text'][i][:100]}...") # 顯示部分輸入
        print(f"  Prediction: {predictions[i]}")
        print(f"  Reference:  {references[i]}")
        
    # 將結果存成 DataFrame
    results_df = pd.DataFrame({
        "predictions": predictions,
        "references": references
    })

    # 儲存為 CSV
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n結果已成功儲存至: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    run_inference()