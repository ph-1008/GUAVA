import pandas as pd
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
MODEL_DIR = r"CODE\Other_task\model_training_mT5\20250526_0353_mt5_prompt_all\best_model"
INPUT_CSV_PATH = os.path.join(MODEL_DIR, "best_train.csv")
OUTPUT_CSV_PATH = os.path.join(MODEL_DIR, "train_set_predictions.csv")
ATTENTION_SAVE_DIR = os.path.join(MODEL_DIR, "attention_matrices")
rcParams['font.family'] = 'Microsoft JhengHei' 
rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
os.makedirs(ATTENTION_SAVE_DIR, exist_ok=True)

def save_attention_heatmap(attention_matrix, index, layer, head, tokens):
    """
    儲存單層單頭的 attention matrix heatmap 圖片
    """
    plt.figure(figsize=(len(tokens) * 0.5, len(tokens) * 0.5))
    sns.heatmap(attention_matrix, cmap="viridis", xticklabels=tokens, yticklabels=tokens)
    plt.title(f"Sample {index} - Layer {layer} Head {head}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = os.path.join(ATTENTION_SAVE_DIR, f"sample{index}_layer{layer}_head{head}.png")
    plt.savefig(filename)
    plt.close()

def run_inference():
    if not os.path.exists(MODEL_DIR):
        print(f"錯誤：找不到模型資料夾 '{MODEL_DIR}'。")
        return
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"錯誤：找不到輸入檔案 '{INPUT_CSV_PATH}'。")
        return

    print(f"正在從 '{MODEL_DIR}' 載入模型和 Tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR, output_attentions=True)
    model.to(device)
    model.eval()

    print(f"正在讀取資料: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)
    df = df.head(10)
    if "input_text" not in df.columns or "target_text" not in df.columns:
        print("錯誤：CSV 缺少必要欄位。")
        return

    predictions = []
    references = []

    print(f"開始對 {len(df)} 筆資料進行推論...")

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Predictions"):
            input_text = row["input_text"]
            target_text = row["target_text"]

            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_length=128,
                output_attentions=True,
                return_dict_in_generate=True
            )

            # 預測文字
            predicted_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            predictions.append(predicted_text)
            references.append(target_text)

            # 處理注意力 (取 encoder attentions)
            # 注意: MT5 generate 的 attention 是解碼器的，需要額外 forward encoder
            encoder_outputs = model.encoder(**inputs)
            attentions = encoder_outputs.attentions  # (num_layers, batch, num_heads, seq_len, seq_len)

            # 解碼 token list（限長以免圖太大）
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            tokens = [t if len(t) < 10 else t[:9]+"…" for t in tokens]

            # 儲存每層每頭的注意力 (可挑幾層幾頭，不必全存)
            for layer in [0, len(attentions)-1]:  # 儲存第一層與最後一層
                for head in [0]:  # 儲存第一個 head
                    attn_matrix = attentions[layer][0, head].detach().cpu().numpy()
                    save_attention_heatmap(attn_matrix, index, layer, head, tokens)

            # 或儲存 numpy matrix 以供後續分析
            # np.save(os.path.join(ATTENTION_SAVE_DIR, f"sample{index}_attn_layer{layer}_head{head}.npy"), attn_matrix)

    print("\n推論完成！")

    print("\n前 5 筆預測 vs. 參考答案:")
    for i in range(min(5, len(predictions))):
        print("-" * 20)
        print(f"  Input:      {df['input_text'][i][:100]}...")
        print(f"  Prediction: {predictions[i]}")
        print(f"  Reference:  {references[i]}")

    results_df = pd.DataFrame({
        "predictions": predictions,
        "references": references
    })

    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n結果已成功儲存至: {OUTPUT_CSV_PATH}")
    print(f"注意力 heatmap 已儲存至資料夾: {ATTENTION_SAVE_DIR}")

if __name__ == "__main__":
    run_inference()
