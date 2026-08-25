import pandas as pd
import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast, Trainer, TrainingArguments # Removed DataCollatorForSeq2Seq import, using custom one
from datasets import Dataset, Features, Value, Sequence
import ast
from sklearn.model_selection import KFold
from rouge import Rouge
import datetime
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

# --- Configuration ---
MODEL_NAME = "google/mt5-base"
DATA_PATH = r"CODE\Other_task\data\data.csv"
MAX_SEQ_LENGTH = 128
POS_EMBEDDING_DIM = 768
PEFT_R = 8
PEFT_LORA_ALPHA = 16
PEFT_LORA_DROPOUT = 0.1
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
RANDOM_STATE = 42
N_SPLITS = 5

# --- Global Tokenizer (Load Once) ---
# Load the FAST version of the tokenizer globally
tokenizer = MT5TokenizerFast.from_pretrained(MODEL_NAME)

# --- Helper Functions ---
def safe_literal_eval(x):
    # ... (definition as before) ...
    if isinstance(x, str):
        try:
            if not all(c in '[],0123456789 ' for c in x):
                 try:
                     return [int(i) for i in x.split()]
                 except:
                    print(f"Error: Could not parse pos_ids string: {x}")
                    return []
            return ast.literal_eval(x)
        except (ValueError, SyntaxError, TypeError):
            print(f"Error evaluating string: {x}. Returning empty list.")
            return []
    elif isinstance(x, (list, tuple)):
        return list(x)
    else:
        print(f"Warning: Unexpected type for pos_ids: {type(x)}. Returning empty list.")
        return []

# --- Moved Preprocessing Function Definition Out of Loop ---
def preprocess_function(examples):
    processed = {
        "input_ids": [], "attention_mask": [], "labels": [], "pos_input_ids": []
    }
    if not all(k in examples for k in ["input_text", "target_text", "pos_ids"]):
        print("Warning: Missing one or more required keys in input batch.")
        return processed

    expected_len = len(examples["input_text"])
    if not (len(examples["target_text"]) == expected_len and len(examples["pos_ids"]) == expected_len):
        print(f"Warning: Mismatched input lengths in batch! Input: {len(examples['input_text'])}, Target: {len(examples['target_text'])}, POS: {len(examples['pos_ids'])}")

    for i in range(expected_len):
        input_text_i = examples["input_text"][i]
        target_text_i = examples["target_text"][i]
        try:
            pos_ids_i = examples["pos_ids"][i]
            if not isinstance(pos_ids_i, list):
                 print(f"Warning: pos_ids at index {i} is not a list (type: {type(pos_ids_i)}). Value: {pos_ids_i}. Using empty list.")
                 pos_ids_i = []
        except IndexError:
            print(f"Warning: IndexError accessing pos_ids at index {i}. Using empty list.")
            pos_ids_i = []

        if not input_text_i or not isinstance(input_text_i, str):
            print(f"Warning: Invalid input_text at index {i}: {input_text_i}. Appending empty lists.")
            processed["input_ids"].append([])
            processed["attention_mask"].append([])
            processed["labels"].append([])
            processed["pos_input_ids"].append([])
            continue

        if not target_text_i or not isinstance(target_text_i, str):
             print(f"Warning: Invalid target_text at index {i}: {target_text_i}. Using empty target.")
             target_text_i = ""

        try:
            # Use global tokenizer
            tokenized_input = tokenizer(input_text_i, truncation=True, max_length=MAX_SEQ_LENGTH)
            tokenized_target = tokenizer(text_target=target_text_i, truncation=True, max_length=MAX_SEQ_LENGTH)
            input_ids = tokenized_input["input_ids"]
            target_ids = tokenized_target["input_ids"]
            original_pos_ids = pos_ids_i

            word_ids = tokenized_input.word_ids()
            if word_ids is None:
                 aligned_pos_ids = [-100] * len(input_ids)
            else:
                 aligned_pos_ids = []
                 for word_idx in word_ids:
                      if word_idx is None:
                          aligned_pos_ids.append(-100)
                      else:
                          if word_idx < len(original_pos_ids) and isinstance(original_pos_ids[word_idx], int):
                               aligned_pos_ids.append(original_pos_ids[word_idx])
                          else:
                               aligned_pos_ids.append(-100)

            if len(aligned_pos_ids) != len(input_ids):
                 print(f"CRITICAL WARNING: Length mismatch input_ids ({len(input_ids)}) vs aligned_pos_ids ({len(aligned_pos_ids)}) idx {i}. Fixing.")
                 aligned_pos_ids.extend([-100] * (len(input_ids) - len(aligned_pos_ids)))
                 aligned_pos_ids = aligned_pos_ids[:len(input_ids)]

            processed["input_ids"].append(input_ids)
            processed["attention_mask"].append(tokenized_input["attention_mask"])
            labels = [(label if label != tokenizer.pad_token_id else -100) for label in target_ids]
            processed["labels"].append(labels)
            processed["pos_input_ids"].append(aligned_pos_ids)

        except Exception as e:
            print(f"ERROR during preprocessing example index {i}: {e}")
            # Append empty lists
            processed["input_ids"].append([])
            processed["attention_mask"].append([])
            processed["labels"].append([])
            processed["pos_input_ids"].append([])

    final_len = len(processed["input_ids"])
    if not (len(processed["attention_mask"]) == final_len and len(processed["labels"]) == final_len and len(processed["pos_input_ids"]) == final_len):
         print(f"CRITICAL WARNING: Mismatched output lengths in final processed batch!")

    return processed


def evaluation_function(model, pos_embedding_layer, val_dataset, tokenizer, device):
    # ... (definition as before, uses global tokenizer if passed) ...
    rouge = Rouge()
    predictions = []
    references = []
    model.eval()
    pos_embedding_layer.eval()
    model.to(device)
    pos_embedding_layer.to(device)
    input_embedding_layer = model.get_input_embeddings()

    with torch.no_grad():
        for example in val_dataset:
            input_text = example["input_text"]
            target_text = example["target_text"]
            # Use the tokenizer passed to the function
            tokenized_input = tokenizer(input_text, return_tensors="pt", padding=False, truncation=True, max_length=MAX_SEQ_LENGTH)
            word_ids = tokenized_input.word_ids(batch_index=0)

            # Safely get original_pos_ids, ensure it's a list from the dict
            original_pos_ids = example.get("pos_ids", []) # Default to empty list if missing
            if not isinstance(original_pos_ids, list):
                 original_pos_ids = [] # Ensure list type

            aligned_pos_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_pos_ids.append(-100)
                else:
                    if word_idx < len(original_pos_ids) and isinstance(original_pos_ids[word_idx], int): # Added int check
                        aligned_pos_ids.append(original_pos_ids[word_idx])
                    else:
                         aligned_pos_ids.append(-100)

            input_ids = tokenized_input["input_ids"].to(device)
            attention_mask = tokenized_input["attention_mask"].to(device)
            pos_ids_tensor = torch.tensor([aligned_pos_ids], dtype=torch.long).to(device)
            word_embeds = input_embedding_layer(input_ids)
            valid_pos_mask = (pos_ids_tensor != -100)
            pos_ids_tensor_safe = pos_ids_tensor.clone()
            pos_ids_tensor_safe[~valid_pos_mask] = 0
            pos_embeds = pos_embedding_layer(pos_ids_tensor_safe)
            pos_embeds[~valid_pos_mask] = 0.0
            inputs_embeds = word_embeds + pos_embeds
            translated = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_length=MAX_SEQ_LENGTH)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            predictions.append(translated_text)
            references.append(target_text)

    if not predictions or not references:
        print("Warning: No predictions or references generated during evaluation.")
        return 0.0
    try:
        scores = rouge.get_scores(predictions, references, avg=True)
        return scores['rouge-l']['f']
    except ValueError as e:
        print(f"Rouge calculation error: {e}")
        print("Predictions:", predictions)
        print("References:", references)
        return 0.0

# --- Custom Data Collator ---
@dataclass
class CustomDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    pos_embedding_layer: Optional[nn.Embedding] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt" # Keep track of desired final tensor type

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        # Override return_tensors if passed explicitly
        current_return_tensors = return_tensors if return_tensors is not None else self.return_tensors

        if not features or not isinstance(features[0], dict): return {}
        if 'pos_input_ids' not in features[0]: raise KeyError("'pos_input_ids' missing")
        if self.pos_embedding_layer is None: raise ValueError("pos_embedding_layer missing")
        if self.model is None: raise ValueError("Model missing")

        target_device = next(self.model.parameters()).device

        # --- Pad Standard Features (input_ids, attention_mask) using tokenizer.pad ---
        # IMPORTANT: Temporarily set return_tensors=None to get lists for length calculation
        keys_to_pad = ['input_ids', 'attention_mask']
        features_for_padding = [{k: feature[k] for k in keys_to_pad if k in feature} for feature in features]

        padded_standard_features_lists = self.tokenizer.pad(
            features_for_padding,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None # Force return lists
        )

        # ---> GET TARGET LENGTH FROM PADDED LIST <---
        if 'input_ids' in padded_standard_features_lists and padded_standard_features_lists['input_ids']:
             # Length of the first padded sequence determines the target length
             target_length = len(padded_standard_features_lists['input_ids'][0])
             print(f"Collator determined target_length from input_ids list: {target_length}")
        else:
             # Fallback or error
             target_length = self.max_length if self.max_length is not None else 0
             if target_length == 0: raise ValueError("Cannot determine padding length.")

        # --- Handle Labels separately (Pad list to target_length) ---
        padded_labels_list = []
        if "labels" in features[0]:
            labels = [feature["labels"] for feature in features]
            # Using target_length derived from input_ids for consistency
            max_label_length = target_length
            padding_side = self.tokenizer.padding_side
            for label_list in labels:
                truncated_label = label_list[:max_label_length]
                remainder_len = max_label_length - len(truncated_label)
                remainder = [self.label_pad_token_id] * remainder_len
                padded_label = truncated_label + remainder if padding_side == "right" else remainder + truncated_label
                padded_labels_list.append(padded_label)
        # --- End Label Handling ---


        # --- Handle pos_input_ids separately (Pad list to target_length) ---
        pos_ids = [feature["pos_input_ids"] for feature in features]
        padded_pos_ids_list = []
        pos_padding_value = -100
        for p in pos_ids:
             truncated_p = p[:target_length]
             pad_len = target_length - len(truncated_p)
             padded_p = truncated_p + [pos_padding_value] * pad_len
             padded_pos_ids_list.append(padded_p)
        # --- End pos_input_ids Handling ---


        # --- CONVERT ALL PADDED LISTS TO TENSORS ON TARGET DEVICE ---
        batch = {}

        # Convert standard padded features (input_ids, attention_mask)
        for key in padded_standard_features_lists:
             if current_return_tensors == "pt":
                  batch[key] = torch.tensor(padded_standard_features_lists[key], dtype=torch.long).to(target_device)
             # Add elif for 'tf' or 'np' if needed
             else:
                  batch[key] = padded_standard_features_lists[key] # Keep as list

        # Convert padded labels
        if padded_labels_list:
            if current_return_tensors == "pt":
                 batch["labels"] = torch.tensor(padded_labels_list, dtype=torch.long).to(target_device)
            else:
                 batch["labels"] = padded_labels_list

        # Convert padded pos_ids (needed for embedding lookup)
        if current_return_tensors == "pt":
             pos_ids_tensor_gpu = torch.tensor(padded_pos_ids_list, dtype=torch.long).to(target_device) if padded_pos_ids_list else torch.empty((len(features), 0), dtype=torch.long).to(target_device)
        else:
             # If not PyTorch, embedding lookup needs adaptation later
             pos_ids_tensor_gpu = padded_pos_ids_list # Store list, but embed lookup expects tensor


        # --- Prepare decoder_input_ids (using tensors) ---
        if hasattr(self.model, "prepare_decoder_input_ids_from_labels") and "labels" in batch:
            # 'labels' is already a tensor on target_device if return_tensors=='pt'
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids # Should already be on correct device
        # --- End decoder_input_ids ---


        # --- Create inputs_embeds (using tensors) ---
        if "input_ids" in batch: # Check if key exists in the final batch dict
            input_ids_gpu = batch.pop("input_ids") # Pop the tensor

            if input_ids_gpu.shape[1] != target_length: raise ValueError(f"Shape mismatch: input_ids")

            word_embedding_layer = self.model.get_input_embeddings()
            word_embeds = word_embedding_layer(input_ids_gpu)

            # pos_ids_tensor_gpu is already prepared and on device
            if pos_ids_tensor_gpu.shape[1] != target_length: raise ValueError(f"Shape mismatch: pos_ids")

            valid_pos_mask = (pos_ids_tensor_gpu != pos_padding_value)
            pos_ids_tensor_safe_gpu = pos_ids_tensor_gpu.clone()
            pos_ids_tensor_safe_gpu[~valid_pos_mask] = 0

            pos_embeds = self.pos_embedding_layer(pos_ids_tensor_safe_gpu)
            pos_embeds[~valid_pos_mask.unsqueeze(-1).expand_as(pos_embeds)] = 0.0

            inputs_embeds = word_embeds + pos_embeds
            batch["inputs_embeds"] = inputs_embeds

            # attention_mask is already a tensor on device in 'batch'
        else:
            print("Warning: 'input_ids' tensor not found for inputs_embeds creation.")

        # --- Final Check (Optional) ---
        # for k, v in batch.items():
        #    if isinstance(v, torch.Tensor) and v.device != target_device:
        #        print(f"Final Check Warning: Tensor '{k}' ended up on {v.device} instead of {target_device}")

        return batch


if __name__ == '__main__':
    # --- Main Execution Logic ---
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_output_dir = f"CODE/Other_task/model_training_mT5/{current_time}_pos_peft" # Added suffix

    # 1. Load Data
    print("Loading dataset...")
    dataset_pd = pd.read_csv(DATA_PATH)
    dataset_pd["pos_ids"] = dataset_pd["pos_ids"].apply(safe_literal_eval)
    required_cols = ["input_text", "target_text", "pos_ids"]
    if not all(col in dataset_pd.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {dataset_pd.columns}. Required: {required_cols}")
    dataset_pd = dataset_pd[required_cols]

    print("Determining POS vocabulary size...")
    all_pos_ids = [pid for sublist in dataset_pd["pos_ids"] for pid in sublist if isinstance(pid, int)]
    if not all_pos_ids:
         num_pos_tags = 1
         max_pos_id = 0 # Set a default if no tags found
         print("Warning: No valid integer pos_ids found. Setting num_pos_tags to 1.")
    else:
        max_pos_id = max(all_pos_ids)
        num_pos_tags = max_pos_id + 1
    print(f"Found {num_pos_tags} unique POS tags (max ID: {max_pos_id}).")

    # 2. K-Fold Setup
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_score = float('-inf')
    best_fold = -1
    all_fold_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- K-Fold Loop ---
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_pd)):
        print(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")
        train_df = dataset_pd.iloc[train_idx].reset_index(drop=True)
        val_df = dataset_pd.iloc[val_idx].reset_index(drop=True)

        features_schema = Features({ # Use schema consistently
            'input_text': Value('string'),
            'target_text': Value('string'),
            'pos_ids': Sequence(Value('int64'))
        })
        train_dataset = Dataset.from_pandas(train_df, features=features_schema)
        val_dataset = Dataset.from_pandas(val_df, features=features_schema)

        print("Mapping preprocessing function (outside loop, no cache)...")
        # --- Apply map with no cache ---
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=False # Disable caching
        )
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=False # Disable caching
        )
        # --- END CHANGE ---

        # --- Check Dataset Features ---
        print(f"Fold {fold+1}: Features in train_dataset AFTER map: {train_dataset.features}")
        print(f"Fold {fold+1}: Features in val_dataset AFTER map: {val_dataset.features}")
        # --- End Check ---


        # Verify necessary columns exist in the features schema
        required_map_cols = ['input_ids', 'attention_mask', 'labels', 'pos_input_ids']
        if not all(col in train_dataset.features for col in required_map_cols):
            raise ValueError(f"Missing required columns in train_dataset features after mapping! Found: {train_dataset.features}")
        if not all(col in val_dataset.features for col in required_map_cols):
             raise ValueError(f"Missing required columns in val_dataset features after mapping! Found: {val_dataset.features}")


        # 4. Model Setup
        print("Loading base model...")
        model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        # model = prepare_model_for_kbit_training(model) # If using k-bit

        print("Creating POS embedding layer...")
        pos_embedding_layer = nn.Embedding(num_pos_tags, POS_EMBEDDING_DIM) # Create on CPU first
        pos_embedding_layer.weight.data.normal_(mean=0.0, std=0.02)
        pos_embedding_layer.to(device) # Move explicitly to device

        print("Configuring PEFT (LoRA)...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=PEFT_R,
            lora_alpha=PEFT_LORA_ALPHA, lora_dropout=PEFT_LORA_DROPOUT,
            target_modules=["q", "v"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # Trainer will move the PEFT model to the device specified in args

        # 5. Training Arguments and Trainer
        fold_output_dir = f"{base_output_dir}/fold_{fold+1}"
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch", save_strategy="epoch",
            learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
            per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS, weight_decay=WEIGHT_DECAY,
            save_total_limit=1, label_smoothing_factor=0.0,
            logging_dir=f"{fold_output_dir}/logs", logging_steps=50,
            remove_unused_columns=False,
            # report_to="tensorboard",
            # fp16=torch.cuda.is_available(), # Re-enable fp16 if desired
            # load_best_model_at_end=True, # Optional
            # metric_for_best_model="eval_loss", # Optional
        )

        data_collator = CustomDataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, pos_embedding_layer=pos_embedding_layer,
            pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
            tokenizer=tokenizer, # Pass tokenizer for potential internal use
            data_collator=data_collator,
        )

        # 6. Train
        print("Starting training...")
        try:
            trainer.train()
        except Exception as train_error:
            print(f"ERROR during training loop: {train_error}")
            # Optionally add more diagnostics here if needed
            raise train_error


        # 7. Evaluate
        print("Evaluating fold model...")
        # Reload best model if load_best_model_at_end=True was used, or use current model
        # If using load_best_model_at_end, the `trainer.model` is already the best one.

        # Ensure model/layer are on eval mode and correct device for eval function
        model.eval()
        pos_embedding_layer.eval()
        model.to(device)
        pos_embedding_layer.to(device)

        # Create list of dicts from the *original* validation split dataframe
        # This ensures eval function gets 'input_text', 'target_text', 'pos_ids' correctly
        val_dataset_list_for_eval = val_df.to_dict('records')

        model_score = evaluation_function(
             model,
             pos_embedding_layer,
             val_dataset_list_for_eval, # Use original data dict
             tokenizer, # Pass tokenizer
             device
        )
        print(f"Fold {fold+1} ROUGE-L F1 Score: {model_score:.4f}")
        all_fold_scores.append(model_score)

        if model_score > best_score:
            best_score = model_score
            best_fold = fold + 1
            print(f"New best fold: {best_fold} with score {best_score:.4f}")
            # Trainer handles saving if save_strategy='epoch' and load_best_model_at_end=True
            # If manual saving needed:
            # best_adapter_path = f"{base_output_dir}/best_adapter_fold_{best_fold}"
            # best_pos_layer_path = f"{base_output_dir}/best_pos_layer_fold_{best_fold}.pt"
            # model.save_pretrained(best_adapter_path) # Saves adapter_config.json, adapter_model.bin
            # torch.save(pos_embedding_layer.state_dict(), best_pos_layer_path)
            # print(f"Saved best adapter to {best_adapter_path}")
            # print(f"Saved best pos layer to {best_pos_layer_path}")


        del model, trainer, pos_embedding_layer # Explicit cleanup
        torch.cuda.empty_cache() # Clear GPU cache

    # --- End of K-Fold Loop ---

    print("\n--- K-Fold Cross Validation Summary ---")
    # ... (Summary printing as before) ...
    if all_fold_scores:
        for i, score in enumerate(all_fold_scores):
            print(f"Fold {i+1} ROUGE-L F1: {score:.4f}")
        average_score = np.mean(all_fold_scores)
        print(f"Average ROUGE-L F1 across {N_SPLITS} folds: {average_score:.4f}")
        if best_fold != -1:
             print(f"Best fold: {best_fold} with ROUGE-L F1: {best_score:.4f}")
    else:
        print("No folds were successfully evaluated.")

    print(f"Training artifacts saved in base directory: {base_output_dir}")
    print("Script finished.")