import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Suppress warnings from datasets library
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import ast
from sklearn.model_selection import KFold
from rouge import Rouge
import datetime
import os
import re
def compress_con_tree(con_str):
    # 移除多層括號，只保留最外層與詞性資訊
    # (X word) -> X:word
    con_str = re.sub(r'\(([^() ]+)\s+([^() ]+)\)', r'(\1:\2)', con_str)  
    con_str = con_str.replace('(', ' ').replace(')', ' ')  # 括號拿掉
    con_str = re.sub(r'\s+', ' ', con_str)  # 多個空格變成一個空格
    return con_str.strip()

def evaluation_function(model, val_dataset, tokenizer, device):
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
    
    print("\nSample Predictions vs References:")
    for i in range(min(5, len(predictions))): print(f"  Pred: {predictions[i]}\n  Ref:  {references[i]}\n" + "-" * 10)
    try:
        filtered_preds = [p.strip().replace('//', '/').replace('/', ' ') for p, r in zip(predictions, references) if p and r]
        filtered_refs = [r.strip().replace('//', '/').replace('/', ' ') for p, r in zip(predictions, references) if p and r]
        if not filtered_preds:
             print("Warning: All prediction/reference pairs were empty after filtering.")
             return 0.0
        rouge_scorer = Rouge() # Instantiate scorer inside function
        scores = rouge_scorer.get_scores(filtered_preds, filtered_refs, avg=True)
        print(f"Evaluation ROUGE-L F1: {scores['rouge-l']['f']:.4f}")
        return scores['rouge-l']['f'], filtered_preds, filtered_refs
    except ValueError as e:
        print(f"Error calculating ROUGE: {e}")
        return 0.0


script_dir = os.path.dirname(os.path.abspath(__file__))


print("hi")
# 讀取資料集
DATASET_PATH = os.path.join("CODE","Other_task", "data", "data.csv") # Adjust as needed
dataset = pd.read_csv(DATASET_PATH)
dataset["pos"] = dataset["pos"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
dataset["dep"] = dataset["dep"].apply(lambda x: ast.literal_eval(x))
dataset['con'] = dataset['con'].apply(lambda x: compress_con_tree(str(x)))
#print(dataset["dep"].head())
# dataset["tok"] = dataset["tok"].apply(lambda x: ast.literal_eval(x))  <POS> {' '.join(row['pos'])}  <DEP> {' '.join([i[1] for i in row['dep']])}   <CON> {row['con']}
dataset["input_text"] = dataset.apply(lambda row: f"translate Chinese to Gloss: {row['input_text']}", axis=1)


if __name__ == "__main__":
    print("go")
    # 取得當前時間
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_output_dir = os.path.join(script_dir, f"{current_time}_MarianMT_prompt_all") # New version ID
    os.makedirs(base_output_dir, exist_ok=True)

    dataset = dataset[["input_text", "target_text"]]
    dataset["target_text"] = dataset["target_text"]#.apply(lambda x: x.replace('//','/').replace('/',','))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = float('-inf')
    best_fold = None
    best_train_df = None
    best_val_df = None
    all_folds_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Helsinki-NLP/opus-mt-zh-en"
        special_tokens_dict = {'additional_special_tokens': ['<POS>', '<DEP>', '<CON>']}
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

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
        
        model = MarianMTModel.from_pretrained(model_name).to(device)
        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold+1}")
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch",
            learning_rate=5e-5,
            warmup_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=40,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=100,
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
        
        model_score , filtered_preds, filtered_refs = evaluation_function(model, val_dataset, tokenizer, device)
        all_folds_scores.append(model_score)

        if model_score > best_score:
            best_score = model_score
            best_fold = fold
            best_model_path = os.path.join(base_output_dir, "best_model")
            print(f"Saving best model to {best_model_path}...")
            trainer.save_model(best_model_path) # Saves the best model loaded at end
            tokenizer.save_pretrained(best_model_path)
            df = pd.DataFrame({"predictions": filtered_preds, "references": filtered_refs})
            df.to_csv(os.path.join(best_model_path, "predictions.csv"), index=False)
            
            # Save corresponding train/val splits
            if train_df is not None and val_df is not None:
                train_df.to_csv(os.path.join(base_output_dir,"best_model", "best_train.csv"), index=False)
                val_df.to_csv(os.path.join(base_output_dir,"best_model", "best_val.csv"), index=False)
        # Clean up GPU memory
        del model, trainer
        torch.cuda.empty_cache()

    if best_fold != -1:
        print(f"Best fold: {best_fold}")
        print(f"Best ROUGE-L F1 score: {best_score:.4f}")
        print(f"Best model saved in: {os.path.join(base_output_dir, 'best_model')}")
        print(f"Best train/val splits saved in: {base_output_dir}")
    else:
        print("No best model was saved (scores might have been 0 or evaluation failed).")

    # Clean up intermediate fold directories (optional)
    print("Cleaning up intermediate fold directories...")
    print(f"ALL score {all_folds_scores}")
    import shutil
    for i in range(1, kf.n_splits + 1):
       fold_dir = os.path.join(base_output_dir, f"fold_{i}")
       if os.path.exists(fold_dir):
           print(f"Removing {fold_dir}...")
           shutil.rmtree(fold_dir, ignore_errors=True)

    print("Script finished.")