import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from datasets import Dataset
from transformers import (
    MT5Tokenizer, MT5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq, AdamW # 引入 AdamW
)
from peft import get_peft_model, LoraConfig, TaskType
# Note: 'adapters.composition as ac' is not used in the provided script.
import os
import datetime
import pandas as pd
from rouge import Rouge
import torch

# ================= PROFILER IMPORTS =================
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader # 需要 DataLoader 來手動迭代
from tqdm import tqdm # 用於顯示進度條
# ====================================================


try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd() # Fallback for notebooks
    print(f"Warning: '__file__' not found. Using current working directory: {script_dir}")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_output_dir = os.path.join(script_dir, current_time) # Store results in a subfolder
os.makedirs(base_output_dir, exist_ok=True)


train_df_path = os.path.join("CODE", "Other_task", "data", "best_train.csv")
eval_df_path = os.path.join("CODE", "Other_task", "data", "best_val.csv")
train_df = pd.read_csv(train_df_path)
eval_df = pd.read_csv(eval_df_path)

train_df = train_df[["input_text", "target_text"]]
eval_df = eval_df[["input_text", "target_text"]]

train_hf_dataset_text = Dataset.from_pandas(train_df)
eval_hf_dataset_text = Dataset.from_pandas(eval_df)


if __name__ == "__main__":
    # ---------------------------
    # 2. Initialize Model & Tokenizer
    # ---------------------------
    model_name = "google/mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': ['<POS>', '<DEP>', '<CON>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # config = LoraConfig(
    #     task_type=TaskType.SEQ_2_SEQ_LM,
    #     inference_mode=False,
    #     r=16,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    # )
    # model = get_peft_model(model, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.print_trainable_parameters()

    # <--- 修改點: 打印整個模型的參數數量 ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---------------------------
    # 3. Data Preprocessing
    # ---------------------------
    max_length = 512

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], max_length=max_length, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"], max_length=128, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        return model_inputs

    cols_to_remove = train_hf_dataset_text.column_names
    
    tokenized_train_dataset = train_hf_dataset_text.map(
        preprocess_function, batched=True, remove_columns=cols_to_remove
    )

    # ---------------------------
    # 4. Profiler & Manual Training Loop Setup
    # ---------------------------
    # 使用與 Trainer 相同的參數
    batch_size = 1 # per_device_train_batch_size
    learning_rate = 1e-4
    
    # 創建 DataLoader
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length")
    train_dataloader = DataLoader(
        tokenized_train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )

    # 創建優化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------
    # 5. Start Profiling and Manual Training for one epoch
    # -----------------------------------------------------
    print("Starting profiling for one epoch...")
    model.train() # 確保模型處於訓練模式
    
    # 初始化 Profiler
    # activities: 指定追蹤 CPU 和 GPU
    # record_shapes: 記錄張量的形狀，有助於分析
    # with_stack: 記錄 Python 調用堆棧，便於追蹤函數來源
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=True # 也可以追蹤記憶體使用
    ) as prof:
        # 手動訓練迴圈
        for step, batch in enumerate(tqdm(train_dataloader, desc="Profiling Epoch 1")):
            # 將數據移到 GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 使用 record_function 來標記程式碼區塊
            # 這樣在報告中會清晰地看到 "forward_pass", "backward_pass", "optimizer_step"
            
            # --- 1. 前向傳播 (Forward Pass) ---
            with record_function("forward_pass"):
                outputs = model(**batch)
                loss = outputs.loss
            
            # --- 2. 反向傳播 (Backward Pass) ---
            with record_function("backward_pass"):
                loss.backward()

            # --- 3. 權重更新 (Optimizer Step) ---
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
            
            # 為了快速測試，可以只跑幾個 step 就中斷
            if step >= 50:
                 break
    
    print("Profiling finished.")

    # ---------------------------
    # 6. Print Profiler Results
    # ---------------------------
    print("\n--- Profiler Results ---")
    
    # 按總時間排序，顯示前 15 個操作
    print("--- Top 15 Events by Self CUDA Time ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    print("\n--- Top 15 Events by Total CUDA Time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # 使用 trace_handler 將結果保存到 JSON 文件，以便用 Chrome Trace Viewer 查看
    trace_file_path = os.path.join(base_output_dir, "profiler_trace.json")
    prof.export_chrome_trace(trace_file_path)
    print(f"\nDetailed trace saved to: {trace_file_path}")
    print("You can open this file in Chrome by navigating to 'chrome://tracing'")


    # ================== 分析我們標記的區塊 ==================
    print("\n--- Analysis of Custom Recorded Functions ---")
    events = prof.key_averages()
    


    # 創建一個字典來保存我們關心的結果
    timing_results = {
        "forward_pass": {"total_time": 0, "count": 0},
        "backward_pass": {"total_time": 0, "count": 0},
        "optimizer_step": {"total_time": 0, "count": 0}
    }
    
    total_epoch_time = 0
    
    for event in events:
        if event.key in timing_results:
            timing_results[event.key]["total_time"] += event.cpu_time_total
            timing_results[event.key]["count"] += event.count
    
    # 計算總時間 (ms)
    for key in timing_results:
        total_epoch_time += timing_results[key]["total_time"]
    
    # 確保我們有數據，避免除以零
    if total_epoch_time > 0:
        # 輸出結果
        print(f"Total time for one epoch (tracked functions): {total_epoch_time / 1000:.2f} ms")
        print("-" * 50)
        
        for name, data in timing_results.items():
            total_ms = data["total_time"] / 1000
            count = data["count"]
            avg_ms = total_ms / count if count > 0 else 0
            percentage = (data["total_time"] / total_epoch_time) * 100 if total_epoch_time > 0 else 0
            
            print(f"Function: {name}")
            print(f"  - Total Time: {total_ms:.2f} ms")
            print(f"  - Average Time per call: {avg_ms:.4f} ms")
            print(f"  - Percentage of Total Time: {percentage:.2f}%")
            print("-" * 20)
    else:
        print("No custom events were recorded. Please check the `record_function` calls.")