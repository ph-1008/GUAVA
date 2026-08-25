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
data_path = os.path.join( "CODE", "Other_task", "data", "data.csv")
if not os.path.exists(data_path):
     # Fallback or alternative path if the original structure doesn't exist
     print(f"Warning: Data file not found at {data_path}. Trying alternative path './data.csv'")
     data_path = os.path.join(script_dir, "data.csv") # Example alternative
     if not os.path.exists(data_path):
         raise FileNotFoundError(f"Could not find data file at {data_path} or the alternative.")

dataset = pd.read_csv(data_path)
dataset["pos"] = dataset["pos"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
dataset["dep"] = dataset["dep"].apply(lambda x: ast.literal_eval(x))
# <POS> {' '.join(row['pos'])}
dataset["input_text"] = dataset.apply(lambda row: f"translate Chinese to Gloss: {row['input_text']} <POS> {' '.join(row['pos'])}", axis=1)
# Clean target text: replace // with / then / with space
dataset["target_text"] = dataset["target_text"].apply(lambda x: x.replace("//", "/").replace("/", " "))
dataset = dataset[["input_text", "target_text"]]

raw_dataset = Dataset.from_pandas(dataset)
dataset_split = raw_dataset.train_test_split(test_size=0.2, seed=42) # Added seed for reproducibility


if __name__ == "__main__":
    # ---------------------------
    # 2. Initialize Model & Tokenizer
    # ---------------------------
    model_name = "google/mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': ['<POS>','<DEP>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
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
    max_length = 128

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
                max_length=max_length,
                truncation=True,
                padding="max_length" # Pad labels to max_length as well
            )
        model_inputs["labels"] = labels["input_ids"]
        # Ensure padding tokens in labels are ignored by the loss function (DataCollator usually handles this)
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        return model_inputs

    tokenized_datasets = dataset_split.map(preprocess_function, batched=True, remove_columns=dataset_split["train"].column_names)

    # ---------------------------
    # 4. Training Settings & Trainer
    # ---------------------------
    training_args = TrainingArguments(
        output_dir=base_output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=50, # Consider reducing epochs for faster testing
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
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

    def predict_gloss(input_str):
        inputs = tokenizer(
            input_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=5,
                num_beams=5,
                early_stopping=True
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Example sentence
    example_input = "translate Chinese to Gloss: 我在樓梯上滑倒了。 <POS> Nh P Na Ncd VA Di PERIODCATEGORY"
    print("\n--- Example Prediction ---")
    predicted_gloss = predict_gloss(example_input)
    print(f"Input: {example_input}")
    print(f"Predicted Gloss: {predicted_gloss}")
    print("------------------------\n")

    # ---------------------------
    # 7. Automatic ROUGE-L F1 Score Calculation (using 'rouge' library)
    # ---------------------------
    def evaluate_rouge_l(dataset_to_eval):
        """Calculates the average ROUGE-L F1 score using the 'rouge' library."""
        rouge = Rouge() # Initialize Rouge object
        all_predictions = []
        all_references = []
        count = 0

        # Use the original split dataset (before tokenization) for text access
        original_test_data = dataset_split["test"] # Reference the split data

        print(f"Generating predictions for ROUGE evaluation on {len(original_test_data)} test samples...")
        for i, sample in enumerate(original_test_data):
            if "input_text" not in sample or "target_text" not in sample:
                print(f"Warning: Skipping sample {i} due to missing keys.")
                continue

            input_text = sample["input_text"]
            reference_text = sample["target_text"]

            # Handle potential empty strings which can cause errors in rouge library
            if not input_text or not reference_text:
                 print(f"Warning: Skipping sample {i} due to empty input ('{input_text}') or reference ('{reference_text}').")
                 continue

            predicted_text = predict_gloss(input_text)

            # Handle cases where prediction might be empty
            if not predicted_text:
                print(f"Warning: Skipping sample {i} due to empty prediction for input: '{input_text}'")
                # Option 1: Skip the sample
                continue
                # Option 2: Assign a default empty string (might slightly lower scores)
                # predicted_text = " " # Or some other placeholder if rouge handles it

            all_predictions.append(predicted_text)
            all_references.append(reference_text)
            count += 1

            if (i + 1) % 50 == 0: # Print progress
                 print(f"Processed {i+1}/{len(original_test_data)} samples...")

        if count == 0:
            print("Warning: No valid samples found for evaluation.")
            return 0.0

        print(f"\nCalculating average ROUGE scores for {count} valid samples...")
        for i in range(min(5, len(all_predictions))): # Print first 5 predictions for debugging
            print(f"Predicted: {all_predictions[i]}")
            print(f"Reference: {all_references[i]}")
            print("-" * 50)
        # Calculate scores in batch using the rouge library
        try:
            # get_scores returns average scores if avg=True
            # The format is {'rouge-1': {'f': ..., 'p': ..., 'r': ...}, 'rouge-2': ..., 'rouge-l': ...}
            scores = rouge.get_scores(all_predictions, all_references, avg=True)
            avg_rouge_l_f1 = scores['rouge-l']['f']
            print("ROUGE calculation complete.")
            return avg_rouge_l_f1
        except ValueError as e:
            print(f"Error during ROUGE calculation: {e}")
            print("This might be due to empty hypothesis or reference strings.")
            # You might want to investigate which specific samples caused this
            # for idx, (p, r) in enumerate(zip(all_predictions, all_references)):
            #     if not p or not r:
            #         print(f"Problematic sample index (approx): {idx}, Pred: '{p}', Ref: '{r}'")
            return 0.0 # Return 0 or raise the error
        except Exception as e:
             print(f"An unexpected error occurred during ROUGE calculation: {e}")
             return 0.0


    # Calculate test set ROUGE-L F1 score
    rouge_l_f1_score = evaluate_rouge_l(dataset_split["test"])
    print(f"\nTest Set Average ROUGE-L F1 Score: {rouge_l_f1_score:.4f}")

    # --- Save results ---
    results = {
        "model_name": model_name,
        "adapter_config": config.to_dict(),
        "training_args": training_args.to_dict(),
        "test_set_size": len(dataset_split["test"]),
        "evaluated_samples_count": len(list(filter(lambda s: s.get("input_text") and s.get("target_text"), dataset_split["test"]))), # Actual count evaluated might differ slightly if skips occurred
        "rouge_l_f1_score": rouge_l_f1_score,
        "example_input": example_input,
        "example_prediction": predicted_gloss,
    }
    results_file = os.path.join(base_output_dir, "results.json")

    # Use a custom encoder to handle non-serializable types like PosixPath
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
                return obj.isoformat()
            elif isinstance(obj, os.PathLike):
                return str(obj)
            # Add other type handlers as needed
            return super(CustomEncoder, self).default(obj)

    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=CustomEncoder, ensure_ascii=False)
        print(f"Results saved to {results_file}")
    except TypeError as e:
         print(f"Error saving results to JSON: {e}")
         print("Attempting to save results by converting problematic types to strings...")
         # Fallback: convert everything potentially problematic to string
         simplified_results = json.loads(json.dumps(results, cls=CustomEncoder, ensure_ascii=False)) # Force serialization/deserialization
         with open(results_file, "w", encoding="utf-8") as f:
             json.dump(simplified_results, f, indent=4, ensure_ascii=False)
         print(f"Simplified results saved to {results_file}")

    # --- Optional: Save the final PEFT model ---
    final_model_path = os.path.join(base_output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model and tokenizer saved to {final_model_path}")