# --- START OF FILE apapter.py ---

from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import pandas as pd
import torch
from datasets import Dataset
import ast
from sklearn.model_selection import KFold
from rouge import Rouge
import datetime
import os
from tqdm import tqdm
import shutil # Added for cleanup

# ===== Custom Adapter Layer =====
# class CustomAdapterLayer(torch.nn.Module):
#     # ... (your CustomAdapterLayer code - unchanged) ...
#     def __init__(self, d_model, pos_dim=64, reduction_factor=16):
#         super().__init__()
#         self.pos_proj = torch.nn.Linear(pos_dim, d_model)
#         self.down_project = torch.nn.Linear(d_model, d_model // reduction_factor)
#         self.activation = torch.nn.SiLU()
#         self.up_project = torch.nn.Linear(d_model // reduction_factor, d_model)
#         self.dropout = torch.nn.Dropout(0.1)

#     def forward(self, hidden_states, pos_ids_embeds):
#         pos = self.pos_proj(pos_ids_embeds)
#         combined = hidden_states + pos
#         down = self.down_project(combined)
#         down = self.activation(down)
#         up = self.up_project(down)
#         return hidden_states + self.dropout(up)


# Adapter with Cross-Attention + Gating
class CustomAdapterLayer(torch.nn.Module):
    def __init__(self, d_model, pos_dim=64, reduction_factor=32, num_heads=4):
        super().__init__()
        self.pos_proj = torch.nn.Linear(pos_dim, d_model)
        self.cross_attn = torch.nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.down_project = torch.nn.Linear(d_model, d_model // reduction_factor)
        self.activation = torch.nn.SiLU()
        self.up_project = torch.nn.Linear(d_model // reduction_factor, d_model)
        self.dropout = torch.nn.Dropout(0.1)
        self.gate = torch.nn.Linear(d_model, 1)  # gating scalar

    def forward(self, hidden_states, pos_ids_embeds, attention_mask=None):
        # pos projection
        pos_embeds = self.pos_proj(pos_ids_embeds)  # (batch, seq, d_model)

        # Cross Attention: token attends to POS info
        attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=pos_embeds,
            value=pos_embeds
        )

        # Combine attention output
        combined = hidden_states + attn_output

        # Bottleneck adapter
        down = self.down_project(combined)
        down = self.activation(down)
        up = self.up_project(down)
        up = self.dropout(up)

        # Gating control
        gate_weight = torch.sigmoid(self.gate(hidden_states))  # (batch, seq, 1)
        output = hidden_states + gate_weight * up

        return output


# ===== Custom Model with POS Embedding =====
class CustomMT5Model(MT5ForConditionalGeneration):
    # ... (your CustomMT5Model code - unchanged) ...
    def __init__(self, config, num_pos_tags=62):
        super().__init__(config)
        self.pos_embedding = torch.nn.Embedding(num_pos_tags, config.d_model)
        torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
        self.pos_embedding.weight.requires_grad = True
        # Initialize adapter, telling it the input POS dimension is d_model
        self.adapter_layer = CustomAdapterLayer(config.d_model, pos_dim=config.d_model) # Pass d_model as pos_dim

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pos_ids=None,
        **kwargs # Capture any other args Trainer might pass
    ):
        # Determine if we need to run the encoder or use provided encoder_outputs
        run_encoder = encoder_outputs is None

        # Calculate combined input embeddings
        if run_encoder and pos_ids is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pos_ids is provided and encoder_outputs is None")
            if inputs_embeds is not None:
                raise ValueError("Cannot provide both input_ids and inputs_embeds when using pos_ids")


            # Step 1: Create embeddings
            token_embeds = self.get_input_embeddings()(input_ids)
            pos_embeds = self.pos_embedding(pos_ids)

            # Step 2: Fuse with Adapter
            fused = self.adapter_layer(token_embeds, pos_embeds)

            inputs_embeds = fused
            input_ids = None # We are now using inputs_embeds
        elif run_encoder and inputs_embeds is not None:
             # If only inputs_embeds are given (no pos_ids scenario)
            input_ids = None

        # Remove unexpected kwargs if Trainer adds them
        kwargs.pop('num_items_in_batch', None)

        # Call the original forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # Pass the potentially modified embeds
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs # Pass the cleaned kwargs
        )

        return outputs


# ===== Evaluation Function =====
def evaluation_function(model, val_dataset, tokenizer, device, max_input_len, max_target_len):
    # ... (your evaluation_function - needs MAX lengths passed or defined globally inside main) ...
    # Make sure MAX_INPUT_LENGTH and MAX_TARGET_LENGTH used inside are accessible
    rouge = Rouge()
    predictions = []
    references = []

    model.to(device)
    model.eval()

    print("Running evaluation...")
    try:
        expected_start_token_id = model.config.decoder_start_token_id
        print(f"Model config expected decoder_start_token_id: {expected_start_token_id} ({tokenizer.decode(expected_start_token_id)})")
    except AttributeError:
        print("model.config.decoder_start_token_id not found, will use tokenizer.pad_token_id")
        expected_start_token_id = None

    decoder_start_id = model.config.pad_token_id
    print(f"Using decoder_start_token_id: {decoder_start_id} ({tokenizer.decode(decoder_start_id)}) for generation.")

    for example in tqdm(val_dataset):
        input_text = example["input_text"]
        target_text = example["target_text"]
        raw_pos_ids = example["pos_ids"]

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_input_len # Use passed max_input_len
        ).to(device)

        input_len = inputs["input_ids"].shape[1]
        if len(raw_pos_ids) < input_len:
            padded_pos_ids = raw_pos_ids + [0] * (input_len - len(raw_pos_ids))
        else:
            padded_pos_ids = raw_pos_ids[:input_len]

        inputs["pos_ids"] = torch.tensor([padded_pos_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pos_ids=inputs["pos_ids"],
                decoder_start_token_id=decoder_start_id,
                max_length=max_target_len, # Use passed max_target_len
                num_beams=4,
                early_stopping=True
            )

        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(translated_text)
        references.append(target_text)

    if not predictions or not references:
        print("Warning: No predictions or references generated for ROUGE calculation.")
        return 0.0

    print("\nSample Predictions vs References:")
    for i in range(min(5, len(predictions))):
        print(f"  Pred: {predictions[i]}")
        print(f"  Ref:  {references[i]}")
        print("-" * 10)

    try:
        filtered_preds = [p.strip().replace('//', '/').replace('/', ' ') for p, r in zip(predictions, references) if p and r]
        filtered_refs = [r.strip().replace('//', '/').replace('/', ' ') for p, r in zip(predictions, references) if p and r]
        if not filtered_preds:
             print("Warning: All prediction/reference pairs were empty after filtering.")
             return 0.0

        scores = rouge.get_scores(filtered_preds, filtered_refs, avg=True)
        print(f"Evaluation ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
        return scores['rouge-l']['f']
    except ValueError as e:
        print(f"Error calculating ROUGE: {e}")
        print("Sample Prediction:", predictions[0] if predictions else "N/A")
        print("Sample Reference:", references[0] if references else "N/A")
        return 0.0


# Define preprocessing function (can be outside main, it's just a definition)
def preprocess_function(examples, tokenizer, max_input_len, max_target_len):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_len, # Use passed arg
        padding="max_length",
        truncation=True
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_len, # Use passed arg
        padding="max_length",
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
        actual_len = sum(attn_mask)
        if len(p_ids) >= actual_len:
            truncated_pos = p_ids[:actual_len]
        else:
            truncated_pos = p_ids + [0] * (actual_len - len(p_ids))

        padded_pos = truncated_pos + [0] * (max_input_len - len(truncated_pos)) # Use passed arg
        processed_pos_ids.append(padded_pos)

    model_inputs["pos_ids"] = processed_pos_ids
    return model_inputs


# ===== Data Collator =====
# (Can be defined outside main)
class DataCollatorWithPos(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model):
        super().__init__(tokenizer, model=model, padding=True)

    def __call__(self, features):
        pos_ids = [f.pop("pos_ids") for f in features]
        batch = super().__call__(features)
        batch["pos_ids"] = torch.tensor(pos_ids, dtype=torch.long)
        return batch


# ==============================================================
#      MAIN EXECUTION BLOCK
# ==============================================================
if __name__ == '__main__':
    # --- Setup ---
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = os.getcwd()
        print(f"Warning: '__file__' not found. Using current working directory: {script_dir}")

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_output_dir = os.path.join(script_dir, f"{current_time}_ada")
    os.makedirs(base_output_dir, exist_ok=True)

    data_path = os.path.join("CODE", "Other_task", "data", "data.csv")
    print(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
         raise FileNotFoundError(f"Data file not found at {data_path}")

    dataset_df = pd.read_csv(data_path) # Renamed to dataset_df to avoid conflict
    required_cols = ["input_text", "target_text", "pos_ids"]
    if not all(col in dataset_df.columns for col in required_cols):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    dataset_df = dataset_df[required_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
    print("Parsing 'pos_ids' column...")
    # Use safer eval or json.loads if possible, but ast is common
    try:
        dataset_df["pos_ids"] = dataset_df["pos_ids"].apply(ast.literal_eval)
        dataset_df["input_text"] = dataset_df["input_text"].apply(lambda x: f"translate Chinese to Gloss:{x}")
        dataset_df["target_text"] = dataset_df["target_text"].apply(lambda x: f"Gloss:{x}".replace("/"," / ").replace("//"," // ").replace("^^"," ^^ "))
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing 'pos_ids': {e}. Check data format.")
        exit() # Or handle more gracefully
    print("Dataset loaded and parsed.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1.0
    best_fold = -1
    # Removed best_train_df, best_val_df definitions here, handled later

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model and Tokenizer ---
    model_name = "google/mt5-base" # <<<====== VERIFY THIS IS THE NAME YOU WANT
    print(f"Attempting to load tokenizer for: {model_name}")
    tokenizer = MT5Tokenizer.from_pretrained(model_name) # Removed target_lang, MT5 doesn't use it like Marian

    # --- Constants ---
    MAX_INPUT_LENGTH = 128
    MAX_TARGET_LENGTH = 128
    NUM_POS_TAGS = 62 # Make sure this matches the highest ID + 1 in your pos_ids data

    # --- K-Fold Cross-Validation ---
    all_fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_df)):
        print(f"\n===== Training Fold {fold+1}/5 =====")
        train_df = dataset_df.iloc[train_idx].reset_index(drop=True)
        val_df = dataset_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        print("Preprocessing training data...")
        train_dataset_processed = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH), # Pass args
            batched=True,
            remove_columns=train_dataset.column_names
        )
        print("Preprocessing validation data...")
        val_dataset_processed = val_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH), # Pass args
            batched=True,
            remove_columns=val_dataset.column_names
        )

        # --- Load Model FOR THIS FOLD ---
        print(f"Loading model: {model_name}")
        # Pass num_pos_tags to the constructor
        model = CustomMT5Model.from_pretrained(model_name, num_pos_tags=NUM_POS_TAGS)

        # --- FREEZE Parameters (Adapter Tuning) ---
        print("Freezing base model parameters...")
        for name, param in model.named_parameters():
            if 'pos_embedding' in name or 'adapter_layer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params} ({(trainable_params / total_params) * 100:.4f}%)")
        # --- End Parameter Freezing ---

        model.to(device) # Move model to device *after* potential freezing

        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold+1}")
        # fold_results_dir = os.path.join(fold_output_dir, "results") # Not used directly

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            # dataloader_num_workers=0, # Set explicitly to 0 for Windows debugging if needed
            eval_strategy="epoch",
            learning_rate=2e-4,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=50, # REDUCED for testing the fix
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", # Use trainer's eval_loss
            greater_is_better=False,
            # logging_dir=os.path.join(fold_output_dir, 'logs'), # Add logging dir
            logging_steps=100, # Log more often
            # Required for gradient checkpointing if used, and good practice
            # remove_unused_columns=False, # Keep if pos_ids needed later? Check Trainer behavior. Usually True.
            label_names=["labels"], # Explicitly tell trainer the label column name
        )

        data_collator = DataCollatorWithPos(tokenizer, model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics, # Define if using evaluate during training
        )

        print("Starting training...")
        trainer.train()

        # --- Evaluation using custom function ---
        # Ensure the best model loaded by trainer is used
        # model = trainer.model # Already done by load_best_model_at_end=True? Reassigning might load from scratch.
        # Let's assume trainer correctly keeps the best model in memory or reloads it.
        model.to(device) # Ensure model is on correct device for evaluation
        model_score = evaluation_function(
            model,
            val_dataset, # Use original (unprocessed) dataset
            tokenizer,
            device,
            MAX_INPUT_LENGTH, # Pass max lengths
            MAX_TARGET_LENGTH
            )
        all_fold_scores.append(model_score)
        print(f"Fold {fold+1} ROUGE-L F1 Score: {model_score:.4f}")

        if model_score > best_score:
            print(f"*** New best score found! Fold {fold+1} ({model_score:.4f} > {best_score:.4f}) ***")
            best_score = model_score
            best_fold = fold + 1
            best_model_path = os.path.join(base_output_dir, "best_model")
            print(f"Saving best model from fold {best_fold} to {best_model_path}...")
            trainer.save_model(best_model_path)
            tokenizer.save_pretrained(best_model_path)

            print(f"Saving best train/val splits from fold {best_fold}...")
            train_df.to_csv(os.path.join(base_output_dir, "best_train.csv"), index=False)
            val_df.to_csv(os.path.join(base_output_dir, "best_val.csv"), index=False)
        else:
             print(f"Score {model_score:.4f} did not improve over best score {best_score:.4f} from fold {best_fold}")

        break
        # Clean up fold directory to save space if not the best fold
        # Note: Best model is saved separately above. `load_best_model_at_end` loads it into memory,
        # but the checkpoint might still be in the fold_output_dir if it was the last one saved.
        # if fold+1 != best_fold: # Be careful if best fold is the last fold
        #     print(f"Cleaning up intermediate directory: {fold_output_dir}")
        #     shutil.rmtree(fold_output_dir, ignore_errors=True)

    # --- Post-Training ---
    print("\n===== Training Complete =====")
    if all_fold_scores:
        print(f"Fold Scores (ROUGE-L F1): {all_fold_scores}")
        print(f"Average ROUGE-L F1 over {kf.n_splits} folds: {sum(all_fold_scores) / len(all_fold_scores):.4f}")
    else:
        print("No fold scores recorded.")

    if best_fold != -1:
        print(f"Best fold: {best_fold}")
        print(f"Best ROUGE-L F1 score: {best_score:.4f}")
        print(f"Best model saved in: {os.path.join(base_output_dir, 'best_model')}")
        print(f"Best train/val splits saved in: {base_output_dir}")
    else:
        print("No best model was saved (scores might have been 0 or evaluation failed).")

    # Optional: Clean up all intermediate fold directories at the end
    print("Cleaning up remaining intermediate fold directories...")
    for i in range(1, kf.n_splits + 1):
        fold_dir = os.path.join(base_output_dir, f"fold_{i}")
        if os.path.exists(fold_dir):
             print(f"Removing {fold_dir}...")
             shutil.rmtree(fold_dir, ignore_errors=True)

    print("Script finished.")

# --- END OF FILE ---