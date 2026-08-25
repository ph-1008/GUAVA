from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import pandas as pd
import os
import glob
from rouge import Rouge

def split_text(data):
    processed_data = [i.strip().replace('//', '/').replace('/', ' ') for i in data]
    # processed_data = []
    # for row in data:
    #     row = row.strip().split("//")
    #     new_row = [' '.join(item.split('/')) for item in row]
    #     processed_data.append(new_row)
    return processed_data


# 獲取當前 Python 檔案所在的資料夾
current_dir = os.path.dirname(os.path.abspath(__file__))

# 找出最新的日期格式資料夾 (格式: YYYYMMDD_HHMM)
model_dirs = sorted(glob.glob(os.path.join(current_dir, "????????_????")), reverse=True)

if not model_dirs:
    raise FileNotFoundError("找不到符合日期格式的模型資料夾！")

latest_model_path = model_dirs[0]  # 取最新的資料夾
print(f"載入模型: {latest_model_path}")

# fold
latest_model_path = os.path.join(latest_model_path,'fold_1/checkpoint-1040')
# 載入微調後的模型
model = T5ForConditionalGeneration.from_pretrained(latest_model_path)
tokenizer = T5Tokenizer.from_pretrained(latest_model_path)

latest_model_path = model_dirs[0]
# 載入測試資料
df = pd.read_csv(os.path.join(latest_model_path, "best_val.csv"))

# 處理翻譯
input_text = df["input_text"]
inputs = tokenizer(input_text.tolist(), return_tensors="pt", padding=True, truncation=True)

# 生成 Gloss
with torch.no_grad():
    translated_tokens = model.generate(**inputs)
    gloss_output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

# 將生成的 Gloss 加入 DataFrame
df["gloss_output"] = gloss_output

sign = split_text(df["target_text"].to_list())
gloss = split_text(df["gloss_output"].to_list())

rouge = Rouge()

output_data = []

for s, g in zip(sign, gloss):
    score = rouge.get_scores(s, g)[0]
    
    rouge_1_f = round(score['rouge-1']['f'], 5)
    rouge_1_p = round(score['rouge-1']['p'], 5)
    rouge_1_r = round(score['rouge-1']['r'], 5)
    rouge_2_f = round(score['rouge-2']['f'], 5)
    rouge_2_p = round(score['rouge-2']['p'], 5)
    rouge_2_r = round(score['rouge-2']['r'], 5)
    rouge_l_f = round(score['rouge-l']['f'], 5)
    rouge_l_p = round(score['rouge-l']['p'], 5)
    rouge_l_r = round(score['rouge-l']['r'], 5)

    output_data.append([rouge_1_f,rouge_1_p,rouge_1_r,rouge_2_f,rouge_2_p,rouge_2_r,rouge_l_f,rouge_l_p,rouge_l_r])

df_rouge = pd.DataFrame(output_data,columns=['rouge_1_f','rouge_1_p','rouge_1_r','rouge_2_f','rouge_2_p','rouge_2_r','rouge_l_f','rouge_l_p','rouge_l_r'])
# 合併 ROUGE 分數到 df

df = pd.concat([df, df_rouge], axis=1)

# 儲存結果為 Excel（包含 gloss_output 和 rouge 分數）
output_path = os.path.join(current_dir, "gloss_output.xlsx")
with pd.ExcelWriter(output_path) as writer:
    df.to_excel(writer, sheet_name="Gloss Output", index=False)

print(f"手語 Gloss 已存成 Excel 檔案: {output_path}")