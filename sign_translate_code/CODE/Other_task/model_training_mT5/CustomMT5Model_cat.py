import torch
from transformers import MT5ForConditionalGeneration, DataCollatorForSeq2Seq
from rouge import Rouge
from tqdm import tqdm

# Define max lengths
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

# ===== Custom Model with POS Embedding (Concatenated) =====
class CustomMT5Model(MT5ForConditionalGeneration):
    def __init__(self, config, num_pos_tags=62):
        super().__init__(config)
        self.pos_embedding = torch.nn.Embedding(num_pos_tags, config.d_model)
        # Initialize POS embeddings (optional but good practice)
        torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
        self.pos_embedding.weight.requires_grad = True

        # --- Add Projection Layer for Concatenated Embeddings ---
        # Input dimension is token_embed_dim + pos_embed_dim = config.d_model + config.d_model
        # Output dimension needs to be config.d_model for the encoder
        self.projection = torch.nn.Linear(config.d_model * 2, config.d_model)
        torch.nn.init.xavier_uniform_(self.projection.weight) # Initialize projection layer
        # --- End Projection Layer ---


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

        # Calculate combined input embeddings using CONCATENATION
        if run_encoder and pos_ids is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pos_ids is provided and encoder_outputs is None")
            if inputs_embeds is not None:
                raise ValueError("Cannot provide both input_ids and inputs_embeds when using pos_ids")

            token_embeds = self.get_input_embeddings()(input_ids)
            pos_embeds = self.pos_embedding(pos_ids) * 0.2 # Scale POS embeddings

            # --- Concatenate and Project ---
            # concatenated_embeds = torch.cat((token_embeds, pos_embeds), dim=-1)
            # inputs_embeds = self.projection(concatenated_embeds)
            # --- End Concatenate and Project ---
            inputs_embeds = token_embeds + pos_embeds # Use addition instead of concatenation
            input_ids = None # We now use inputs_embeds for the encoder
        elif run_encoder and inputs_embeds is not None:
            # If inputs_embeds are provided directly (without pos_ids), use them
            input_ids = None
        # If encoder_outputs are provided, inputs_embeds are not needed for the encoder

        # Clean kwargs potentially passed by Trainer that are not expected by base MT5
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
            inputs_embeds=inputs_embeds, # Pass the potentially modified inputs_embeds
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs # Pass the cleaned kwargs
        )

        return outputs

# ===== Evaluation Function (Unchanged) =====
def evaluation_function(model, val_dataset, tokenizer, device):
    rouge = Rouge()
    predictions = []
    references = []

    model.to(device)
    model.eval()

    print("Running evaluation...")
    # --- Check the ID the model expects ---
    try:
        expected_start_token_id = model.config.decoder_start_token_id
        print(f"Model config expected decoder_start_token_id: {expected_start_token_id} ({tokenizer.decode(expected_start_token_id)})")
    except AttributeError:
        print("model.config.decoder_start_token_id not found, will use tokenizer.pad_token_id")
        expected_start_token_id = None # Fallback below

    # Use the model's pad token ID as the start token ID if not explicitly set otherwise
    # For MT5, pad_token_id (0) is often used implicitly if decoder_start_token_id isn't set correctly during fine-tuning.
    # Let's stick to the model's config default or pad_token_id if necessary.
    decoder_start_id = model.config.decoder_start_token_id if model.config.decoder_start_token_id is not None else tokenizer.pad_token_id
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
                decoder_start_token_id=decoder_start_id, # Explicitly set
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
        return scores['rouge-l']['f'],filtered_preds,filtered_refs
    except ValueError as e:
        print(f"Error calculating ROUGE: {e}")
        print("Sample Prediction:", predictions[0] if predictions else "N/A")
        print("Sample Reference:", references[0] if references else "N/A")
        return 0.0 # Return a default score on error

# Data Collator (Unchanged)
class DataCollatorWithPos(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model):
        super().__init__(tokenizer, model=model, padding=True)

    def __call__(self, features):
        pos_ids = [f.pop("pos_ids") for f in features]
        batch = super().__call__(features)
        batch["pos_ids"] = torch.tensor(pos_ids, dtype=torch.long)
        return batch

