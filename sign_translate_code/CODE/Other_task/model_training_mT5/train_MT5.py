import pandas as pd
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import ast
from sklearn.model_selection import KFold
from rouge import Rouge
import datetime

def evaluation_function(model, val_dataset, tokenizer, device):
    rouge = Rouge()
    predictions = []
    references = []
    
    model.to(device)
    
    for example in val_dataset:
        input_text = example["input_text"]
        target_text = example["target_text"]
        
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        predictions.append(translated_text)
        references.append(target_text)
    
    scores = rouge.get_scores(predictions, references, avg=True)
    return scores['rouge-l']['f']

# 取得當前時間
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_output_dir = f"CODE/Other_task/model_training_T5_chinese/{current_time}"

# 讀取資料集
dataset = pd.read_csv(r"CODE\Other_task\data\data.csv")
# # 讀取資料集
# dataset = pd.read_csv(r"CODE/Other_task/multi_model/data.csv")
# dataset["pos"] = dataset["pos"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# dataset["input_text"] = dataset.apply(lambda row: f"translate Chinese to Gloss: {row['input_text']} <POS> {' '.join(row['pos'])}", axis=1)

dataset = dataset[["input_text", "target_text"]]
dataset["target_text"] = dataset["target_text"]#.apply(lambda x: x.replace('//','/').replace('/',','))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = float('-inf')
best_fold = None
best_train_df = None
best_val_df = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold+1}...")
    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    val_df = dataset.iloc[val_idx].reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
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
    
    model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    training_args = TrainingArguments(
        output_dir=f"CODE/Other_task/multi_model/results_fold_{fold}",
        eval_strategy="epoch",
        learning_rate=5e-5,
        warmup_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        save_strategy="best",
        save_total_limit=1,
        label_smoothing_factor=0.0,
        logging_dir=f"CODE/Other_task/multi_model/logs_fold_{fold}",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    trainer.train()
    
    model_score = evaluation_function(model, val_dataset, tokenizer, device)
    
    if model_score > best_score:
        best_score = model_score
        best_fold = fold
        best_train_df = train_df
        best_val_df = val_df

if best_train_df is not None and best_val_df is not None:
    best_train_df.to_csv(f"CODE/Other_task/multi_model/best_train.csv", index=False)
    best_val_df.to_csv(f"CODE/Other_task/multi_model/best_val.csv", index=False)
    print(f"最佳的 fold 為 {best_fold+1}，已保存最佳的 train/val 數據集。")
