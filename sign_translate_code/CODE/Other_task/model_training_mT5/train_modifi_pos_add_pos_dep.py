# -*- coding: utf-8 -*-
# --- START OF FILE train_modifi_pos_add_pos_dep.py ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Suppress warnings from datasets library

# ---> IMPORT TokenizerFast and logging <---
from transformers import MT5TokenizerFast, Trainer, TrainingArguments
import pandas as pd
import torch
from datasets import Dataset, Features, Value, Sequence
import ast
import datetime
import os
from CustomMT5Model_add_dep_pos import CustomMT5Model, DataCollatorWithPos, evaluation_function, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH,align_tags_to_subtokens2 # Import necessities
from sklearn.model_selection import KFold
import json # For saving mappings



# ===== Configuration & Setup =====
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd() # Fallback for notebooks
    print(f"Warning: '__file__' not found. Using current working directory: {script_dir}")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_output_dir = os.path.join(script_dir, f"{current_time}_mt5_custom_pos_dep_head_v2") # New version ID
os.makedirs(base_output_dir, exist_ok=True)
print(f"Output directory: {base_output_dir}")

# Define paths for the specific train/validation files
DATASET_PATH = os.path.join("CODE","Other_task", "data", "data.csv") # Adjust as needed

# Model settings
MODEL_NAME = "google/mt5-base" # or "google/mt5-base" or "google/mt5-large"
NUM_POS_TAGS = 62

# Training settings
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.01
KFOLD_SPLITS = 5
TASK_PREFIX = "translate Chinese to Gloss: "
TASK_PREFIX = ""
# ===== Load and Prepare Data =====
print(f"Loading dataset from: {DATASET_PATH}")
try:
    dataset_df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    exit()

# Select columns
dataset_df = dataset_df[["input_text", "target_text", "pos", "pos_ids", "dep","tok"]]

# --- Safely evaluate string representations ---
def safe_literal_eval(val):
    # ... (keep safe_literal_eval as before) ...
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not evaluate string: {val}. Returning empty list.")
            return []
    elif isinstance(val, list):
        return val
    else:
        return []

print("Parsing 'pos_ids' column...")
dataset_df['pos_ids'] = dataset_df['pos_ids'].apply(safe_literal_eval)
# also parse the original 'pos' string list if needed elsewhere
dataset_df['pos'] = dataset_df['pos'].apply(safe_literal_eval)

print("Parsing 'tok' column and renaming to ckip_tokens...")
dataset_df['tok'] = dataset_df['tok'].apply(safe_literal_eval)
dataset_df.rename(columns={'tok': 'ckip_tokens'}, inplace=True)


print("Parsing 'dep' column and extracting head/dep_label info...")
dep_label_map = {"<pad>": 0}
current_id = 1

# ... (keep parse_dependencies function as before) ...
def parse_dependencies(dep_str, ckip_token_list):
    # ... function code ...
    global current_id, dep_label_map # Allow modification of the global map/counter
    num_ckip_tokens = len(ckip_token_list)
    head_ids_orig = list(range(num_ckip_tokens))
    dep_label_ids_orig = [dep_label_map["<pad>"]] * num_ckip_tokens
    dep_info = safe_literal_eval(dep_str)
    if not isinstance(dep_info, list):
        warnings.warn(f"Unexpected dependency format after eval: {dep_info}. Returning defaults.")
        return head_ids_orig, dep_label_ids_orig
    num_dep_relations = len(dep_info)
    process_len = num_ckip_tokens
    if num_dep_relations != num_ckip_tokens:
        warnings.warn(f"Mismatch between tokens ({num_ckip_tokens}) and deps ({num_dep_relations}). Processing min.")
        process_len = min(num_ckip_tokens, num_dep_relations)
    for i in range(process_len):
        item = dep_info[i]
        if not isinstance(item, tuple) or len(item) != 2:
            warnings.warn(f"Malformed dependency item at index {i}: {item}. Skipping.")
            continue
        head_idx_1based, label = item
        if not isinstance(head_idx_1based, int):
             warnings.warn(f"Non-integer head index '{head_idx_1based}' at {i}. Skipping.")
             continue
        if head_idx_1based == 0: head_idx_0based = i
        else: head_idx_0based = head_idx_1based - 1
        if not (0 <= head_idx_0based < num_ckip_tokens):
             warnings.warn(f"Head index {head_idx_0based} out of bounds at {i}. Defaulting to self-loop.")
             head_idx_0based = i
        head_ids_orig[i] = head_idx_0based
        if not isinstance(label, str):
             warnings.warn(f"Non-string label '{label}' at {i}. Using padding label.")
             label = "<pad>"
        if label not in dep_label_map:
            dep_label_map[label] = current_id
            current_id += 1
        dep_label_ids_orig[i] = dep_label_map[label]
    return head_ids_orig, dep_label_ids_orig

# Apply parsing using ckip_tokens
parsed_deps_list = [parse_dependencies(row['dep'], row['ckip_tokens'])
                    for _, row in dataset_df.iterrows()]
dataset_df['head_ids_orig'] = [item[0] for item in parsed_deps_list]
dataset_df['dep_label_ids_orig'] = [item[1] for item in parsed_deps_list]

NUM_DEP_TAGS = len(dep_label_map)
print(f"Found {NUM_DEP_TAGS} unique dependency tags.")
print(f"Dependency label map (sample): {dict(list(dep_label_map.items())[:10])}...")
dep_map_path = os.path.join(base_output_dir, "dep_label_map.json")
with open(dep_map_path, 'w') as f: json.dump(dep_label_map, f)
print(f"Dependency label map saved to {dep_map_path}")

# --- Drop processed raw columns ---
print("Dropping processed raw columns: 'pos' list string, 'dep' string")
dataset_df = dataset_df.drop(columns=['pos', 'dep'])

# --- Define Dataset Features ---
print("Defining dataset features...")
dataset_features = Features({
    'input_text': Value('string'),
    'target_text': Value('string'),
    'pos_ids': Sequence(Value('int64')),
    'head_ids_orig': Sequence(Value('int64')),
    'dep_label_ids_orig': Sequence(Value('int64')),
    'ckip_tokens': Sequence(Value('string')),
})
print("Columns in DataFrame before mapping:", dataset_df.columns.tolist())

# ===== Tokenizer =====
print(f"Loading FAST tokenizer: {MODEL_NAME}")
# ---> Use TokenizerFast <---
tokenizer = MT5TokenizerFast.from_pretrained(MODEL_NAME) # legacy=False is often implicit/default for Fast

# ===== Preprocessing Function (No Manual Padding for Aligned Tags) =====
def preprocess_function(examples):
    # Add task prefix HERE
    inputs_with_prefix = [TASK_PREFIX + text for text in examples["input_text"]]

    # Tokenize inputs WITHOUT padding
    inputs_unpadded = tokenizer(
        inputs_with_prefix,
        max_length=MAX_INPUT_LENGTH, # Still truncate
        padding=False,               # DO NOT PAD HERE
        truncation=True,
        return_attention_mask=False  # Not needed yet
    )

    # Tokenize targets WITH padding (standard practice)
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True
    )
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    # --- Store UNPADDED aligned tags ---
    batch_unpadded_aligned_pos_ids = []
    batch_unpadded_aligned_head_ids = []
    batch_unpadded_aligned_dep_ids = []

    pos_pad_id = 0
    head_pad_id = 0 
    dep_pad_id = dep_label_map["<pad>"]

    prefix_tokens = tokenizer(TASK_PREFIX, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_tokens)

    for i in range(len(examples["input_text"])):
        ckip_tokens = examples["ckip_tokens"][i]
        ckip_pos_ids = examples["pos_ids"][i]
        ckip_head_ids = examples["head_ids_orig"][i]
        ckip_dep_label_ids = examples["dep_label_ids_orig"][i]
        mt5_input_ids_with_prefix = inputs_unpadded["input_ids"][i]
        mt5_input_ids_for_alignment = mt5_input_ids_with_prefix[prefix_len:]

        if mt5_input_ids_with_prefix[:prefix_len] != prefix_tokens:
             warnings.warn(f"Prefix tokenization mismatch for example {i}. Alignment might be inaccurate.")

        # --- Perform Alignment ---
        aligned_pos = align_tags_to_subtokens2(ckip_tokens, ckip_pos_ids, mt5_input_ids_for_alignment, tokenizer, pad_tag_id=pos_pad_id)
        aligned_head = align_tags_to_subtokens2(ckip_tokens, ckip_head_ids, mt5_input_ids_for_alignment, tokenizer, pad_tag_id=head_pad_id)
        aligned_dep = align_tags_to_subtokens2(ckip_tokens, ckip_dep_label_ids, mt5_input_ids_for_alignment, tokenizer, pad_tag_id=dep_pad_id)

        # --- DEBUG CHECK (still useful) ---
        expected_align_len = len(mt5_input_ids_for_alignment)
        if len(aligned_pos) != expected_align_len:
            warnings.warn(f"[Example {i}] POS Alignment length mismatch! Forcing length.")
            aligned_pos = (aligned_pos + [pos_pad_id] * expected_align_len)[:expected_align_len]
        if len(aligned_head) != expected_align_len:
             warnings.warn(f"[Example {i}] Head Alignment length mismatch! Forcing length.")
             aligned_head = (aligned_head + [head_pad_id] * expected_align_len)[:expected_align_len]
        if len(aligned_dep) != expected_align_len:
             warnings.warn(f"[Example {i}] Dep Alignment length mismatch! Forcing length.")
             aligned_dep = (aligned_dep + [dep_pad_id] * expected_align_len)[:expected_align_len]

        # --- Combine prefix padding with aligned tags (NO end padding yet) ---
        prefix_padding_pos = [pos_pad_id] * prefix_len
        unpadded_aligned_pos = prefix_padding_pos + aligned_pos

        prefix_padding_head = [head_pad_id] * prefix_len
        unpadded_aligned_head = prefix_padding_head + aligned_head
        # Ensure head IDs are valid indices *before* collating
        unpadded_aligned_head = [min(max(h, 0), MAX_INPUT_LENGTH - 1) for h in unpadded_aligned_head]


        prefix_padding_dep = [dep_pad_id] * prefix_len
        unpadded_aligned_dep = prefix_padding_dep + aligned_dep

        # Append UNPADDED results
        batch_unpadded_aligned_pos_ids.append(unpadded_aligned_pos)
        batch_unpadded_aligned_head_ids.append(unpadded_aligned_head)
        batch_unpadded_aligned_dep_ids.append(unpadded_aligned_dep)


    # --- Return UNPADDED aligned tags along with UNPADDED inputs ---
    # The DataCollator will handle padding input_ids, attention_mask, AND our custom tags
    model_inputs = {
        # Pass unpadded input_ids; collator will pad them
        "input_ids": inputs_unpadded["input_ids"],
        # Labels are already padded, keep them as is
        "labels": labels["input_ids"],
        # Pass unpadded aligned tags; collator needs to pad these
        "pos_ids": batch_unpadded_aligned_pos_ids,
        "head_ids": batch_unpadded_aligned_head_ids,
        "dep_label_ids": batch_unpadded_aligned_dep_ids,
    }
    return model_inputs

# ===== K-Fold Cross-Validation =====
kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
fold_results = []
best_overall_score = -1.0
best_model_path = None
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_df)):
        print(f"\n===== Starting Fold {fold+1}/{KFOLD_SPLITS} =====")
        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        train_df = dataset_df.iloc[train_idx].reset_index(drop=True)
        val_df = dataset_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets objects
        train_dataset = Dataset.from_pandas(train_df, features=dataset_features)
        val_dataset = Dataset.from_pandas(val_df, features=dataset_features)

        # Apply preprocessing
        print("Preprocessing training data (using new alignment)...")
        train_dataset_processed = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=list(dataset_features.keys()) # Remove all original columns
        )
        print("Preprocessing validation data (using new alignment)...")
        val_dataset_processed = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=list(dataset_features.keys())
        )

        # Set format for PyTorch
        train_dataset_processed.set_format("torch")
        val_dataset_processed.set_format("torch")

        # Load model for this fold
        print("Loading model...")
        model = CustomMT5Model.from_pretrained(
            MODEL_NAME,
            num_pos_tags=NUM_POS_TAGS,
            num_dep_tags=NUM_DEP_TAGS
        ).to(device)

        # Define Training Arguments for this fold
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # report_to="tensorboard", # Optional
            # fp16=torch.cuda.is_available(), # Optional
        )

    # --- Inside the KFold loop ---
        # Instantiate the data collator (pass tokenizer, other args optional if defaults are okay)
        data_collator = DataCollatorWithPos(tokenizer=tokenizer) # Removed model=model from here if base class doesn't need it for padding logic

        # Initialize Trainer (rest is the same)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            tokenizer=tokenizer,
            data_collator=data_collator, # Pass the updated collator instance
        )
    #------------------------------

        # Start Training
        print(f"Starting training for fold {fold+1}...")
        trainer.train()
        print(f"Training finished for fold {fold+1}.")

        # --- Post-Training Evaluation for this Fold ---
        print(f"\n===== Evaluating Fold {fold+1} =====")
        print("Evaluating the best model from this fold on the validation set...")
        model.to(device)
        # NOTE: evaluation_function needs alignment if passing tags to generate.
        # Assuming evaluation_function is modified or doesn't need tags for generation.
        final_score,filtered_preds,filtered_refs = evaluation_function(
            model,
            val_dataset,      # Use original val_dataset
            tokenizer,
            device,
            max_input_length=MAX_INPUT_LENGTH,   # Pass the constant
            max_target_length=MAX_TARGET_LENGTH, # Pass the constant
            task_prefix=TASK_PREFIX            # Pass the constant
        )
        fold_results.append(final_score)

        print(f"\nFold {fold+1} ROUGE-L F1 Score on Validation Set: {final_score:.4f}")
        df = pd.DataFrame({"predictions": filtered_preds, "references": filtered_refs})
        df.to_csv(os.path.join(fold_output_dir, "predictions.csv"), index=False)
        # Save the best model for this fold
        fold_best_model_path = os.path.join(fold_output_dir, "best_model")
        print(f"Saving the best model for fold {fold+1} to {fold_best_model_path}...")
        trainer.save_model(fold_best_model_path)
        tokenizer.save_pretrained(fold_best_model_path)
        print(f"Best model and tokenizer for fold {fold+1} saved.")

        # Track overall best model
        if final_score > best_overall_score:
            print(f"New best overall score found: {final_score:.4f} (previous: {best_overall_score:.4f})")
            best_overall_score = final_score
            best_model_path = fold_best_model_path

        # Clean up GPU memory
        del model, trainer
        torch.cuda.empty_cache()
        #exit()

    # ===== Final Results =====
    print("\n===== K-Fold Cross-Validation Complete =====")
    for i, score in enumerate(fold_results): print(f"Fold {i+1} ROUGE-L F1: {score:.4f}")
    average_rouge_l = sum(fold_results) / len(fold_results) if fold_results else 0
    print(f"\nAverage ROUGE-L F1 across {KFOLD_SPLITS} folds: {average_rouge_l:.4f}")
    print(f"Best overall ROUGE-L F1 score: {best_overall_score:.4f}")
    if best_model_path: print(f"The best performing model was saved in: {best_model_path}")
    else: print("Could not determine the best model path.")

    print("Script finished.")
    # --- END OF FILE train_modifi_pos_add_pos_dep.py ---