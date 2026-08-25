import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import re
from sklearn.model_selection import train_test_split
# 定義清理函數
def clean_text(item):
    if not isinstance(item, str):  # 確保是字串
        return item
    item = item.replace("(", "").replace(")", "")  # 移除小括號
    item = re.sub(r'\{.*?\}', '', item)  # 移除大括號內的內容
    item = re.sub(r'\[.*?\]', '', item)  # 移除中括號內的內容
    return item.strip()  # 移除前後空白

# 1. 載入數據集
dataset = pd.read_excel(r"output2.xlsx")
dataset = dataset[dataset['是否採用'] == 1].reset_index(drop=True)
dataset = dataset[['口語', '手語']].rename(columns={'口語': 'input_text', '手語': 'target_text'})
dataset['target_text'] = dataset['target_text'].apply(clean_text)

# 80% 訓練，20% 驗證
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset["input_text"], dataset["target_text"], test_size=0.2, random_state=42
)


train_df = pd.DataFrame({"input_text": train_texts, "target_text": train_labels})
val_df = pd.DataFrame({"input_text": val_texts, "target_text": val_labels})
train_df.to_csv(r"CODE\Other_task\model_training\train.csv", index=False)
val_df.to_csv(r"CODE\Other_task\model_training\val.csv", index=False)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 2. 載入 MarianMT 模型 & Tokenizer
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 3. 數據預處理（Tokenization）
def preprocess_function(examples):
    inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in target] 
        for target in targets["input_ids"]
    ]
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 4. 設定訓練參數
training_args = TrainingArguments(
    output_dir=r"CODE\Other_task\model_training\results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    warmup_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# 5. 開始訓練
trainer.train()

# 6. 儲存微調後的模型
model.save_pretrained(r"CODE\Other_task\model_training\fine_tuned_model")
tokenizer.save_pretrained(r"CODE\Other_task\model_training\fine_tuned_model")