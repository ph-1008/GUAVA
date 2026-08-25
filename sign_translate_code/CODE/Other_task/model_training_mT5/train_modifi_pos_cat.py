# -*- coding: utf-8 -*-
# --- START OF FILE train_modifi.py ---

from transformers import  MT5Tokenizer, Trainer, TrainingArguments
import pandas as pd
import torch
from datasets import Dataset
import ast
import datetime
import os
from CustomMT5Model_cat import *

# ===== Dataset Load & Train Setup =====
# Ensure base_output_dir is correctly defined
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd() # Fallback for notebooks
    print(f"Warning: '__file__' not found. Using current working directory: {script_dir}")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_output_dir = os.path.join(script_dir, f"{current_time}_cat") # Single output directory
os.makedirs(base_output_dir, exist_ok=True)

# Define paths for the specific train/validation files
train_data_path = os.path.join("CODE", "Other_task", "data", "best_train.csv")
val_data_path = os.path.join("CODE", "Other_task", "data", "best_val.csv")

print(f"Loading training dataset from: {train_data_path}")
if not os.path.exists(train_data_path):
     raise FileNotFoundError(f"Training data file not found at {train_data_path}")
train_df = pd.read_csv(train_data_path)

print(f"Loading validation dataset from: {val_data_path}")
if not os.path.exists(val_data_path):
     raise FileNotFoundError(f"Validation data file not found at {val_data_path}")
val_df = pd.read_csv(val_data_path)

# Ensure required columns exist
required_cols = ["input_text", "target_text", "pos_ids"]
if not all(col in train_df.columns for col in required_cols):
    raise ValueError(f"Training Dataset must contain columns: {required_cols}")
if not all(col in val_df.columns for col in required_cols):
    raise ValueError(f"Validation Dataset must contain columns: {required_cols}")

train_df = train_df[required_cols]
val_df = val_df[required_cols]

print("Parsing 'pos_ids' column for training data...")
train_df["pos_ids"] = train_df["pos_ids"].apply(ast.literal_eval)
print("Parsing 'pos_ids' column for validation data...")
val_df["pos_ids"] = val_df["pos_ids"].apply(ast.literal_eval)
print("Datasets loaded and parsed.")

# --- No KFold needed ---
# kf = KFold(n_splits=5, shuffle=True, random_state=42) # REMOVED
# best_score = -1.0 # REMOVED
# best_fold = -1 # REMOVED
# best_train_df = None # REMOVED
# best_val_df = None # REMOVED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name, target_lang ='zh')

# Define preprocessing function (Unchanged)
def preprocess_function(examples):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        padding="max_length", # Pad now for training/eval datasets
        truncation=True
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        text_target=examples["target_text"], # Use text_target for labels
        max_length=MAX_TARGET_LENGTH,
        padding="max_length", # Pad now for training/eval datasets
        truncation=True
    )

    # Set -100 for padding tokens in labels
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    # --- Handle POS IDs ---
    processed_pos_ids = []
    for p_ids, attn_mask in zip(examples["pos_ids"], model_inputs["attention_mask"]):
        actual_len = sum(attn_mask) # Get actual length before padding
        if len(p_ids) >= actual_len:
            truncated_pos = p_ids[:actual_len]
        else:
            truncated_pos = p_ids + [0] * (actual_len - len(p_ids))

        # Pad the truncated sequence to MAX_INPUT_LENGTH
        padded_pos = truncated_pos + [0] * (MAX_INPUT_LENGTH - len(truncated_pos))
        processed_pos_ids.append(padded_pos)

    model_inputs["pos_ids"] = processed_pos_ids
    return model_inputs



# --- Single Training Run ---
print("\n===== Starting Training Run =====")

# Create Datasets objects from loaded dataframes
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df) # Keep original val_dataset for final evaluation

# Apply preprocessing
print("Preprocessing training data...")
train_dataset_processed = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)
print("Preprocessing validation data...")
val_dataset_processed = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

# Load model
print("Loading model...")
model = CustomMT5Model.from_pretrained(model_name).to(device)

# Define Training Arguments for the single run
training_args = TrainingArguments(
    output_dir=base_output_dir, # Main output directory
    # dataloader_num_workers = 4, # Uncomment if needed and workers available
    eval_strategy="epoch",       # Evaluate at the end of each epoch
    learning_rate=3e-5,
    warmup_steps=100,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,   # Adjust based on GPU memory
    num_train_epochs=40,            # Set desired number of epochs
    weight_decay=0.01,
    save_strategy="epoch",          # Save checkpoint at the end of each epoch
    save_total_limit=1,             # Only keep the best checkpoint
    load_best_model_at_end=True,    # Load the best model (lowest eval loss) when training ends
    metric_for_best_model="eval_loss", # Metric to determine the best model
    greater_is_better=False,        # Lower eval_loss is better
    # logging_dir=os.path.join(base_output_dir, 'logs'), # Directory for logs
    # logging_steps=100,              # Log training loss every N steps
    # report_to="tensorboard",        # Log to tensorboard (optional)
)

# Instantiate the data collator
data_collator = DataCollatorWithPos(tokenizer, model)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_processed,
    eval_dataset=val_dataset_processed, # Use processed data for trainer's internal eval
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start Training
print("Starting training...")
trainer.train()

# --- Post-Training Evaluation and Saving ---
print("\n===== Training Complete =====")

# The trainer automatically loads the best model if load_best_model_at_end=True
# We can now evaluate this best model using our custom evaluation function
print("Evaluating the best model on the validation set...")
model.to(device) # Ensure model is on the correct device after training
final_score = evaluation_function(model, val_dataset, tokenizer, device) # Use original val_dataset

print(f"\nFinal ROUGE-L F1 Score on Validation Set: {final_score:.4f}")

# Save the final best model and tokenizer
final_model_path = os.path.join(base_output_dir, "best_model")
print(f"Saving the best model to {final_model_path}...")
trainer.save_model(final_model_path) # Saves the best model loaded at the end
tokenizer.save_pretrained(final_model_path)
print(f"Best model and tokenizer saved in: {final_model_path}")

print("Script finished.")
# --- END OF FILE train_modifi.py ---