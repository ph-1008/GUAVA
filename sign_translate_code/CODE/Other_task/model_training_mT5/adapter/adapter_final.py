# ✅ MT5 + Adapter + POS Prompt 微調整合版
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from datasets import Dataset
from transformers import (
    MT5Tokenizer, MT5ForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
# Note: 'adapters.composition as ac' is not used in the provided script.
import os
import datetime
import ast
import pandas as pd
# --- ROUGE Import ---
# Make sure to install the library: pip install rouge
from rouge import Rouge # Using the 'rouge' library now
# --------------------
import torch
import json # For saving results
import re

def compress_con_tree(con_str):
    # 移除多層括號，只保留最外層與詞性資訊
    # (X word) -> X:word
    con_str = re.sub(r'\(([^() ]+)\s+([^() ]+)\)', r'(\1:\2)', con_str)  
    con_str = con_str.replace('(', ' ').replace(')', ' ')  # 括號拿掉
    con_str = re.sub(r'\s+', ' ', con_str)  # 多個空格變成一個空格
    return con_str.strip()
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd() # Fallback for notebooks
    print(f"Warning: '__file__' not found. Using current working directory: {script_dir}")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_output_dir = os.path.join(script_dir, current_time) # Store results in a subfolder
os.makedirs(base_output_dir, exist_ok=True)

# --- Data Loading and Preprocessing ---
# Ensure the path to your data is correct
# data_path = os.path.join( "CODE", "Other_task", "data", "data.csv")
# if not os.path.exists(data_path):
#      # Fallback or alternative path if the original structure doesn't exist
#      print(f"Warning: Data file not found at {data_path}. Trying alternative path './data.csv'")
#      data_path = os.path.join(script_dir, "data.csv") # Example alternative
#      if not os.path.exists(data_path):
#          raise FileNotFoundError(f"Could not find data file at {data_path} or the alternative.")

# dataset = pd.read_csv(data_path)
# dataset["pos"] = dataset["pos"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# dataset["dep"] = dataset["dep"].apply(lambda x: ast.literal_eval(x))
# dataset['con'] = dataset['con'].apply(lambda x: compress_con_tree(str(x)))
# #print(dataset["dep"].head())
# # dataset["tok"] = dataset["tok"].apply(lambda x: ast.literal_eval(x))  <POS> {' '.join(row['pos'])}  <DEP> {' '.join([i[1] for i in row['dep']])}   <CON> {row['con']}
# dataset["input_text"] = dataset.apply(lambda row: f"translate Chinese to Gloss: {row['input_text']}", axis=1)

# Use more descriptive names for DataFrames
train_df_path = os.path.join("CODE", "Other_task", "data", "best_train.csv")
eval_df_path = os.path.join("CODE", "Other_task", "data", "best_val.csv")
train_df = pd.read_csv(train_df_path)
eval_df = pd.read_csv(eval_df_path)

# Apply any necessary text cleaning if not already done in CSVs
# train_df["target_text"] = train_df["target_text"].apply(lambda x: x.replace("//", "/").replace("/", " "))
# eval_df["target_text"] = eval_df["target_text"].apply(lambda x: x.replace("//", "/").replace("/", " "))

train_df = train_df[["input_text", "target_text"]]
eval_df = eval_df[["input_text", "target_text"]]

# Convert to Hugging Face Datasets (these will retain original text columns for evaluation)
train_hf_dataset_text = Dataset.from_pandas(train_df)
eval_hf_dataset_text = Dataset.from_pandas(eval_df)

#raw_dataset = Dataset.from_pandas(dataset)
#dataset_split = raw_dataset.train_test_split(test_size=0.2, seed=42) # Added seed for reproducibility


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
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        # target_modules=["q", "v"] # Optional: Specify target modules
    )
    model = get_peft_model(model, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.print_trainable_parameters()
    # ---------------------------
    # 3. Data Preprocessing
    # ---------------------------
    max_length = 512

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_length,
            truncation=True,
            padding="max_length" # Pad to max_length during preprocessing
        )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=128,
                truncation=True,
                padding="max_length" # Pad labels to max_length as well
            )
        model_inputs["labels"] = labels["input_ids"]
        # Ensure padding tokens in labels are ignored by the loss function (DataCollator usually handles this)
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        return model_inputs

    cols_to_remove = train_hf_dataset_text.column_names # e.g. ['input_text', 'target_text', '__index_level_0__']
    
    tokenized_train_dataset = train_hf_dataset_text.map(
        preprocess_function,
        batched=True,
        remove_columns=cols_to_remove
    )
    tokenized_val_dataset = eval_hf_dataset_text.map(
        preprocess_function,
        batched=True,
        remove_columns=cols_to_remove # Assumes eval_hf_dataset_text has same column names
    )


    # ---------------------------
    # 4. Training Settings & Trainer
    # ---------------------------
    training_args = TrainingArguments(
        output_dir=base_output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=40, # Consider reducing epochs for faster testing
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir=os.path.join(base_output_dir, 'logs'),
        logging_steps=100,
        save_strategy="epoch",
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=100,
        max_grad_norm=1.0,
        # fp16=torch.cuda.is_available(), # Enable mixed precision if GPU supports it
        # report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ---------------------------
    # 5. Start Training
    # ---------------------------
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # ---------------------------
    # 6. Inference Test Function
    # ---------------------------
    # Load best model implicitly if load_best_model_at_end=True
    model.eval() # Set model to evaluation mode


    # --- Evaluation Function (BLEU & ROUGE) ---
    def evaluate_metrics(dataset_with_text, model_to_eval, tokenizer_for_eval, device_for_eval):
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth_func = SmoothingFunction().method1
        rouge_scorer = Rouge() # Initialize Rouge object
        
        all_predictions = []
        all_references = []

        print("\nGenerating predictions for evaluation...")
        for example in dataset_with_text: # Iterates over HF Dataset with text columns
            input_text = example["input_text"]
            target_text = example["target_text"]
            
            # Use the same generation logic as predict_gloss for consistency
            tokenized_inputs = tokenizer_for_eval(
                input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device_for_eval)
            
            with torch.no_grad():
                translated_ids = model_to_eval.generate(
                    input_ids=tokenized_inputs["input_ids"],
                    attention_mask=tokenized_inputs["attention_mask"],
                    max_length=128, # Max generation length
                    num_beams=5,
                    early_stopping=True,
                    min_length=5 # Optional: consistent with predict_gloss
                )
            translated_text = tokenizer_for_eval.decode(translated_ids[0], skip_special_tokens=True)
            
            all_predictions.append(translated_text)
            all_references.append(target_text)
        valid_predictions = [p.strip().replace('//', '/').replace('/', ' ') for p, r in zip(all_predictions, all_references) if p and r]
        valid_references = [r.strip().replace('//', '/').replace('/', ' ') for p, r in zip(all_predictions, all_references) if p and r]
        print("\nSample Predictions vs References:")
        for i in range(min(5, len(valid_predictions))):
            print(f"  Pred: {valid_predictions[i]}\n  Ref:  {valid_references[i]}\n" + "-" * 10)

        bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores = [], [], [], []
        for pred, ref in zip(valid_predictions, valid_references):
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            if not pred_tokens: pred_tokens = [""] # Handle empty prediction for BLEU

            bleu_1_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1, 0, 0, 0)))
            bleu_2_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.5, 0.5, 0, 0)))
            bleu_3_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1/3, 1/3, 1/3, 0)))
            bleu_4_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.25, 0.25, 0.25, 0.25)))

        avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0
        avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores) if bleu_2_scores else 0
        avg_bleu_3 = sum(bleu_3_scores) / len(bleu_3_scores) if bleu_3_scores else 0
        avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores) if bleu_4_scores else 0
        
        # ROUGE scores
        # Filter out empty predictions for ROUGE, as it can cause errors.
        # Or ensure predictions are never empty (e.g., model always outputs something, even if just EOS).


        if valid_predictions and valid_references:
            rouge_scores = rouge_scorer.get_scores(valid_predictions, valid_references, avg=True)
            rouge_l_f1 = rouge_scores['rouge-l']['f']
            rouge_1_f1 = rouge_scores['rouge-1']['f']
        else:
            rouge_l_f1 = 0
            rouge_1_f1 = 0

        print(f"Avg BLEU-1: {avg_bleu_1:.4f}")
        print(f"Avg BLEU-2: {avg_bleu_2:.4f}")
        print(f"Avg BLEU-3: {avg_bleu_3:.4f}")
        print(f"Avg BLEU-4: {avg_bleu_4:.4f}")
        print(f"ROUGE-L F1: {rouge_l_f1:.4f}")
        print(f"ROUGE-1 F1: {rouge_1_f1:.4f}")

        return { # Return metrics for potential logging
            "bleu1": avg_bleu_1, "bleu2": avg_bleu_2, "bleu3": avg_bleu_3, "bleu4": avg_bleu_4,
            "rougeL_f1": rouge_l_f1, "rouge1_f1": rouge_1_f1
        }

    print("\n--- Final Evaluation on Validation Set (using eval_hf_dataset_text) ---")
    # Use eval_hf_dataset_text, which contains the original 'input_text' and 'target_text'
    evaluation_results = evaluate_metrics(eval_hf_dataset_text, model, tokenizer, device)
    print(f"Evaluation Results: {evaluation_results}")


    final_model_path = os.path.join(base_output_dir, "final_model_peft") # Clarify it's PEFT adapters
    # trainer.save_model(final_model_path) # Trainer already saves best model based on metric_for_best_model
    # If you want to save the one currently in memory (which should be the best one if load_best_model_at_end=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final PEFT model adapters and tokenizer saved to {final_model_path}")