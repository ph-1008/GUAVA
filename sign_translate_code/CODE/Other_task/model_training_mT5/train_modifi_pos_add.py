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
# ===== Custom Model with POS Embedding =====
class CustomMT5Model(MT5ForConditionalGeneration):
    def __init__(self, config, num_pos_tags=62):
        super().__init__(config)
        self.pos_embedding = torch.nn.Embedding(num_pos_tags, config.d_model)
        torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
        self.pos_embedding.weight.requires_grad = True

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

            token_embeds = self.get_input_embeddings()(input_ids)
            pos_embeds = self.pos_embedding(pos_ids)
            inputs_embeds = token_embeds + pos_embeds
            input_ids = None
        elif run_encoder and inputs_embeds is not None:
            input_ids = None

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
            inputs_embeds=inputs_embeds,
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
def evaluation_function(model, val_dataset, tokenizer, device):
    rouge = Rouge()
    predictions = []
    references = []

    model.to(device)
    model.eval()

    print("Running evaluation...")
    # --- Check the ID the model expects ---
    # It's usually the pad_token_id for MarianMT
    try:
        expected_start_token_id = model.config.decoder_start_token_id
        print(f"Model config expected decoder_start_token_id: {expected_start_token_id} ({tokenizer.decode(expected_start_token_id)})")
    except AttributeError:
        print("model.config.decoder_start_token_id not found, will use tokenizer.pad_token_id")
        expected_start_token_id = None # Fallback below

    # Use the model's pad token ID as the start token ID if not explicitly set otherwise
    decoder_start_id = model.config.pad_token_id
    print(f"Using decoder_start_token_id: {decoder_start_id} ({tokenizer.decode(decoder_start_id)}) for generation.")
    # --- End Check ---

    for example in tqdm(val_dataset): # Add tqdm progress bar
        input_text = example["input_text"]
        target_text = example["target_text"]
        raw_pos_ids = example["pos_ids"]

        # Tokenize input text ON THE FLY for evaluation
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=False, # Keep this False for on-the-fly evaluation
            truncation=True,
            max_length=MAX_INPUT_LENGTH # Use same max_length as training input
        ).to(device)

        # Ensure pos_ids match the tokenized input_ids length for THIS example
        input_len = inputs["input_ids"].shape[1]
        if len(raw_pos_ids) < input_len:
            padded_pos_ids = raw_pos_ids + [0] * (input_len - len(raw_pos_ids)) # Use 0 for padding POS tag
        else:
            padded_pos_ids = raw_pos_ids[:input_len]

        inputs["pos_ids"] = torch.tensor([padded_pos_ids], dtype=torch.long).to(device)

        # Use model generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pos_ids=inputs["pos_ids"],
                # --- SOLUTION: Explicitly set the decoder start token ID ---
                decoder_start_token_id=decoder_start_id,
                # --- End Solution ---
                max_length=MAX_TARGET_LENGTH, # Set max length for output
                num_beams=4,          # Beam search often improves quality
                early_stopping=True
            )

        # Decode output
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predictions.append(translated_text)
        references.append(target_text)


    if not predictions or not references:
        print("Warning: No predictions or references generated for ROUGE calculation.")
        return 0.0 # Return a default score

    print("\nSample Predictions vs References:")
    for i in range(min(5, len(predictions))):
        print(f"  Pred: {predictions[i]}")
        print(f"  Ref:  {references[i]}")
        print("-" * 10)

    # Calculate ROUGE-L
    try:
        # Filter out empty predictions/references which cause Rouge errors
        filtered_preds = [p.strip().replace('//', '/').replace('/', ' ') for p, r in zip(predictions, references) if p and r]
        filtered_refs = [r.strip().replace('//', '/').replace('/', ' ') for p, r in zip(predictions, references) if p and r]
        if not filtered_preds:
             print("Warning: All prediction/reference pairs were empty after filtering.")
             return 0.0

        scores = rouge.get_scores(filtered_preds, filtered_refs, avg=True)
        print(f"Evaluation ROUGE-L F1: {scores['rouge-l']['f']:.4f}") # Print score
        return scores['rouge-l']['f']
    except ValueError as e:
        print(f"Error calculating ROUGE: {e}")
        print("Sample Prediction:", predictions[0] if predictions else "N/A")
        print("Sample Reference:", references[0] if references else "N/A")
        return 0.0 # Return a default score on error


# ===== Dataset Load & Train Loop =====
# Ensure base_output_dir is correctly defined if __file__ is not available (e.g., in notebooks)
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd() # Fallback for notebooks
    print(f"Warning: '__file__' not found. Using current working directory: {script_dir}")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
base_output_dir = os.path.join(script_dir, current_time) # Store results in a subfolder
os.makedirs(base_output_dir, exist_ok=True) # Create directory if needed

# Define data path relative to script location or provide absolute path
data_path = os.path.join("CODE", "Other_task", "data", "data.csv") # Adjust if needed
# data_path = r"C:\path\to\your\CODE\Other_task\data\data.csv" # Or use absolute path

print(f"Loading dataset from: {data_path}")
if not os.path.exists(data_path):
     raise FileNotFoundError(f"Data file not found at {data_path}")

dataset = pd.read_csv(data_path)
# Ensure required columns exist
required_cols = ["input_text", "target_text", "pos_ids"]
if not all(col in dataset.columns for col in required_cols):
    raise ValueError(f"Dataset must contain columns: {required_cols}")

dataset = dataset[required_cols]
print("Parsing 'pos_ids' column...")
dataset["pos_ids"] = dataset["pos_ids"].apply(ast.literal_eval)
print("Dataset loaded and parsed.")


kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = -1.0 # Initialize with a valid score like 0 or -1
best_fold = -1
best_train_df = None
best_val_df = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name, target_lang ='zh')

# Define max lengths
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256

# Define preprocessing function outside the loop
def preprocess_function(examples):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input_text"],
        # source_lang = "zh",
        max_length=MAX_INPUT_LENGTH,
        padding="max_length", # Pad now for training/eval datasets
        truncation=True
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        text_target=examples["target_text"], # Use text_target for labels
        max_length=MAX_TARGET_LENGTH,
        # target_lang = "zh",
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
    # Truncate or pad pos_ids to match MAX_INPUT_LENGTH
    processed_pos_ids = []
    for p_ids, attn_mask in zip(examples["pos_ids"], model_inputs["attention_mask"]):
        actual_len = sum(attn_mask) # Get actual length before padding
        if len(p_ids) >= actual_len:
            truncated_pos = p_ids[:actual_len]
        else:
            # This case shouldn't happen if POS tagging matches tokenization
            # but handle defensively: pad with 0 (assuming 0 is a safe pad ID)
            truncated_pos = p_ids + [0] * (actual_len - len(p_ids))
            # print(f"Warning: POS sequence shorter than tokenized input for: {examples['input_text'][:50]}...")


        # Pad the truncated sequence to MAX_INPUT_LENGTH
        padded_pos = truncated_pos + [0] * (MAX_INPUT_LENGTH - len(truncated_pos))
        processed_pos_ids.append(padded_pos)

    model_inputs["pos_ids"] = processed_pos_ids
    return model_inputs


# Data Collator - Now simpler as padding is done in preprocess
class DataCollatorWithPos(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model):
        super().__init__(tokenizer, model=model, padding=True) # Use base class padding

    def __call__(self, features):
        # Separate pos_ids before calling the base collator
        pos_ids = [f.pop("pos_ids") for f in features]

        # Let the base DataCollatorForSeq2Seq handle padding/batching for other keys
        batch = super().__call__(features)

        # Add pos_ids back, converting to tensor. They should already be padded correctly.
        batch["pos_ids"] = torch.tensor(pos_ids, dtype=torch.long)
        return batch


# --- Training Loop ---
all_fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n===== Training Fold {fold+1}/5 =====")
    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    val_df = dataset.iloc[val_idx].reset_index(drop=True)

    # Create Datasets objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df) # Keep original val_dataset for evaluation_function

    # Apply preprocessing
    print("Preprocessing training data...")
    train_dataset_processed = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    print("Preprocessing validation data...")
    # Preprocess val_dataset for trainer's internal evaluation
    val_dataset_processed = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    # Load model for this fold
    print("Loading model...")
    model = CustomMT5Model.from_pretrained(model_name).to(device)

    fold_output_dir = os.path.join(base_output_dir, f"fold_{fold+1}")
    fold_results_dir = os.path.join(fold_output_dir, "results") # Specific subdir for results

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        # dataloader_num_workers = 4,
        eval_strategy="epoch",
        learning_rate=5e-5, 
        warmup_steps=100,
        per_device_train_batch_size=8, # Adjust based on GPU memory
        per_device_eval_batch_size=8, # Can often be larger for eval
        num_train_epochs=40, # Keep it short for testing, increase later (e.g., 3-5)
        weight_decay=0.01,
        save_strategy="epoch", # Save at the end of each epoch
        save_total_limit=1, 
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
        greater_is_better=False, 
    )

    # Instantiate the data collator
    data_collator = DataCollatorWithPos(tokenizer, model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=val_dataset_processed, # Use processed data for trainer eval
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    model.to(device)
    # Use the original val_dataset (not processed/padded) for evaluation_function
    model_score = evaluation_function(model, val_dataset, tokenizer, device)
    all_fold_scores.append(model_score)
    print(f"Fold {fold+1} ROUGE-L F1 Score: {model_score:.4f}")

    # --- Check if this is the best fold ---
    if model_score > best_score:
        print(f"*** New best score found! Fold {fold+1} ({model_score:.4f} > {best_score:.4f}) ***")
        best_score = model_score
        best_fold = fold + 1 # Use 1-based index for reporting
        # Save the best model explicitly to the base output directory
        best_model_path = os.path.join(base_output_dir, "best_model")
        print(f"Saving best model to {best_model_path}...")
        trainer.save_model(best_model_path) # Saves the best model loaded at end
        tokenizer.save_pretrained(best_model_path)

        # Save corresponding train/val splits
        if train_df is not None and val_df is not None:
             train_df.to_csv(os.path.join(base_output_dir, "best_train.csv"), index=False)
             val_df.to_csv(os.path.join(base_output_dir, "best_val.csv"), index=False)
    else:
        print(f"Score {model_score:.4f} did not improve over best score {best_score:.4f}")


# --- Post-Training ---
print("\n===== Training Complete =====")
print(f"Fold Scores (ROUGE-L F1): {all_fold_scores}")
print(f"Average ROUGE-L F1 over {kf.n_splits} folds: {sum(all_fold_scores) / len(all_fold_scores):.4f}")

if best_fold != -1:
    print(f"Best fold: {best_fold}")
    print(f"Best ROUGE-L F1 score: {best_score:.4f}")
    print(f"Best model saved in: {os.path.join(base_output_dir, 'best_model')}")
    print(f"Best train/val splits saved in: {base_output_dir}")
else:
    print("No best model was saved (scores might have been 0 or evaluation failed).")

# Clean up intermediate fold directories (optional)
print("Cleaning up intermediate fold directories...")
import shutil
for i in range(1, kf.n_splits + 1):
    fold_dir = os.path.join(base_output_dir, f"fold_{i}")
    if os.path.exists(fold_dir):
         print(f"Removing {fold_dir}...")
         shutil.rmtree(fold_dir, ignore_errors=True)

print("Script finished.")