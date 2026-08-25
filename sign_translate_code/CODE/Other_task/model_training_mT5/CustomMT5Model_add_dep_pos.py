# --- START OF FILE CustomMT5Model_add_dep_pos.py ---

import torch
from transformers import MT5ForConditionalGeneration, DataCollatorForSeq2Seq, PreTrainedTokenizerFast # Added PreTrainedTokenizerFast
from rouge import Rouge
from tqdm import tqdm
import torch.nn.functional as F # Import F
import warnings

# Define max lengths (Make sure these are accessible or defined here/passed)
# If they are defined in the training script, you might need to pass them
MAX_INPUT_LENGTH = 128 # Example
MAX_TARGET_LENGTH = 128 # Example
# TASK_PREFIX = "translate Chinese to Gloss: " # Example

# Assume align_tags_to_subtokens is also defined/imported here
# If it's in the training script, it needs to be moved here or imported.
# Let's define it here for completeness:
# (Copy the align_tags_to_subtokens function definition here)
# ===== Alignment Function (NEW VERSION) =====
def align_tags_to_subtokens(
    ckip_tokens: list[str],
    ckip_tags: list[int],
    mt5_input_ids: list[int],
    tokenizer: PreTrainedTokenizerFast,
    pad_tag_id: int = -100
) -> list[int]:
    """
    Aligns word-level tags (from CKIP) to subword tokens (from MT5).

    Handles cases where MT5 tokenization is finer (e.g., '翻譯' -> ' 翻', '譯')
    and coarser (e.g., '很', '好' -> ' 很好') than CKIP.

    When MT5 is finer, assigns the original tag to *all* subtokens corresponding
    to the original CKIP word.
    When MT5 is coarser, the single MT5 subtoken gets the tag of the *first*
    CKIP token it represents.

    Assumes `mt5_input_ids` are generated without special tokens like <s>, </s>,
    but *may* include the leading space (' ') token typical of SentencePiece models.
    The function handles this by typically padding the tag at index 0 if it's
    the space token and starting the main alignment from index 1.

    Args:
        ckip_tokens: List of word tokens from CKIP.
        ckip_tags: List of tags corresponding to ckip_tokens.
        mt5_input_ids: List of input IDs from the MT5 tokenizer for the *same* text
                       (typically generated with add_special_tokens=False).
        tokenizer: The (fast) Hugging Face tokenizer instance (e.g., MT5TokenizerFast).
        pad_tag_id: ID for padding subtokens or unaligned tokens. Defaults to -100.

    Returns:
        A list of tag IDs aligned with mt5_input_ids.
    """
    if len(ckip_tokens) != len(ckip_tags):
        raise ValueError("Length of ckip_tokens and ckip_tags must be the same.")

    # Handle empty input case gracefully
    if not mt5_input_ids:
            #  log.warning("MT5 input IDs are empty, but CKIP tokens exist. Returning empty alignment.")
        return []
    if not ckip_tokens:
        # log.warning("CKIP tokens are empty. Returning all pad tags.")
        return [pad_tag_id] * len(mt5_input_ids)


    mt5_subtokens = tokenizer.convert_ids_to_tokens(mt5_input_ids)
    aligned_tags = [pad_tag_id] * len(mt5_input_ids)

    # log.debug(f"Starting alignment: CKIP Tokens={ckip_tokens} (Tags={ckip_tags}), MT5 Subtokens={mt5_subtokens}")

    ckip_idx = 0
    # Start matching from the first *actual* subtoken (index 1), assuming index 0
    # is the SentencePiece leading space ' ' which should remain padded.
    # If the input *doesn't* have a leading space, this will skip the first token.
    # Consider adjusting if input format guarantees no leading space.
    mt5_idx = 1 # Start from the token *after* the potential leading space

    # Handle the case where the input is just the space token
    if len(mt5_subtokens) == 1 and mt5_subtokens[0] == ' ':
        #  log.debug("Input contains only the space token. Returning padded list.")
         return aligned_tags # Already all pad_tag_id

    while ckip_idx < len(ckip_tokens) and mt5_idx < len(mt5_subtokens):
        ckip_token = ckip_tokens[ckip_idx]
        ckip_tag = ckip_tags[ckip_idx]
        # log.debug(f"\nProcessing CKIP Token #{ckip_idx}: '{ckip_token}' (Tag: {ckip_tag}) | Starting at MT5 Index {mt5_idx}")

        # Handle potential empty strings from CKIP if they occur
        if not ckip_token:
            #  log.debug(f"  Skipping empty CKIP token at index {ckip_idx}")
             ckip_idx += 1
             continue

        current_reconstruction = ""
        subtoken_indices_in_match_attempt = [] # All mt5 indices involved in this attempt

        temp_mt5_idx = mt5_idx
        match_found = False # Flag to indicate if a match occurred in the inner loop

        while temp_mt5_idx < len(mt5_subtokens):
            subtoken = mt5_subtokens[temp_mt5_idx]
            # Robust cleaning: remove prefix space marker ONLY if it exists
            cleaned_subtoken = subtoken.replace(' ', '') if subtoken.startswith(' ') else subtoken
            # Handle cases where cleaning results in empty (e.g. subtoken was just ' ')
            # Should not happen with typical SentencePiece if not at start, but safety check.
            if not cleaned_subtoken and subtoken:
                 cleaned_subtoken = ''

            # log.debug(f"  Considering MT5 Subtoken #{temp_mt5_idx}: '{subtoken}' | Cleaned: '{cleaned_subtoken}'")

            subtoken_indices_in_match_attempt.append(temp_mt5_idx)

            potential_reconstruction = current_reconstruction + cleaned_subtoken
            # log.debug(f"    Potential Reconstruction: '{potential_reconstruction}'")

            # --- Alignment Check 1: CKIP token starts with reconstruction (MT5 finer/equal) ---
            if ckip_token.startswith(potential_reconstruction):
                # log.debug(f"    Match Type 1: CKIP '{ckip_token}' starts with Rec '{potential_reconstruction}'")
                current_reconstruction = potential_reconstruction

                # If perfect match for CKIP token
                if current_reconstruction == ckip_token:
                    # log.debug(f"    >>> Full Match (Type 1) Found for '{ckip_token}' <<<")
                    # log.debug(f"    Assigning tag {ckip_tag} to indices {subtoken_indices_in_match_attempt}")
                    # *** MODIFICATION START ***
                    # Assign the tag to *all* subtokens involved in this match
                    for idx in subtoken_indices_in_match_attempt:
                        # Check if tag already assigned (e.g. by a previous coarser match)
                        # Only overwrite if it's currently a pad tag.
                        if aligned_tags[idx] == pad_tag_id:
                           aligned_tags[idx] = ckip_tag
                        # else:
                           # This case implies a previous Type 2 match assigned a tag here.
                           # Keep the tag assigned by the Type 2 match (first CKIP word).
                        #    log.debug(f"    Index {idx} already has tag {aligned_tags[idx]} (likely from Type 2 coarser match), keeping it instead of {ckip_tag}.")
                    # *** MODIFICATION END ***

                    ckip_idx += 1 # Move to next CKIP token
                    mt5_idx = temp_mt5_idx + 1 # Commit consumption, start next search after current subtoken
                    match_found = True
                    # log.debug(f"    Advancing CKIP to {ckip_idx}, MT5 to {mt5_idx}. Aligned: {aligned_tags}")
                    break # Exit inner loop for this CKIP token

                # Else: Partial match, continue consuming MT5 tokens
                temp_mt5_idx += 1
                continue

            # --- Alignment Check 2: Reconstruction starts with CKIP token (MT5 coarser) ---
            # This check is relevant if the *first* subtoken being considered itself
            # contains the entire CKIP token (e.g., MT5 ' 很好' vs CKIP '很').
            elif len(subtoken_indices_in_match_attempt) == 1 and \
                 potential_reconstruction.startswith(ckip_token):
                 # Condition: Only one subtoken considered so far for this word, and it contains the CKIP token.
                #  log.debug(f"    Match Type 2: Rec '{potential_reconstruction}' starts with CKIP '{ckip_token}'")

                 current_subtoken_idx = subtoken_indices_in_match_attempt[0]
                 # Assign tag to this single MT5 subtoken *only if it's not already tagged*
                 if aligned_tags[current_subtoken_idx] == pad_tag_id:
                    #  log.debug(f"    Assigning tag {ckip_tag} to index {current_subtoken_idx}")
                     aligned_tags[current_subtoken_idx] = ckip_tag
                #  else:
                     # If the coarse token already has a tag (from the *first* CKIP token it matched), keep it.
                    #  log.debug(f"    Index {current_subtoken_idx} already has tag {aligned_tags[current_subtoken_idx]}, keeping it (for CKIP token '{ckip_token}').")

                 # Crucially: Only advance CKIP index. MT5 index stays the same (mt5_idx),
                 # allowing the *next* CKIP token (e.g., '好') to check against the *same* coarse MT5 token (' 很好').
                 ckip_idx += 1
                 # mt5_idx remains unchanged for the next outer loop iteration
                 match_found = True
                #  log.debug(f"    Advancing CKIP to {ckip_idx}. MT5 index remains {mt5_idx}. Aligned: {aligned_tags}")
                 break # Exit inner loop, proceed to next CKIP token with same MT5 start index

            # --- Mismatch ---
            else:
                # This occurs if ckip_token.startswith(potential_reconstruction) was false,
                # AND potential_reconstruction.startswith(ckip_token) was false (or len > 1).
                # This means the current subtoken fundamentally doesn't align.
                # log.warning(
                #     f"Mismatch: Cannot align CKIP token '{ckip_token}' (index {ckip_idx}) "
                #     f"starting with MT5 subtoken '{mt5_subtokens[mt5_idx]}' (index {mt5_idx}). "
                #     f"CKIP: '{ckip_token}', Potential Rec from first subtoken: '{potential_reconstruction}'. "
                #     f"Strategy: Skip CKIP token, retry from same MT5 position ({mt5_idx})."
                # )
                # We don't pad here because no match was made involving these indices yet.
                # The indices will be re-evaluated for the next CKIP token from the same mt5_idx.

                ckip_idx += 1 # Skip this CKIP token
                # mt5_idx remains unchanged, allowing next CKIP token to try from the same spot
                match_found = True # Set flag to prevent outer 'else' failure log
                # log.debug(f"    Advancing CKIP to {ckip_idx}. MT5 index remains {mt5_idx}. Aligned: {aligned_tags}")
                break # Exit inner loop, try next CKIP token from same MT5 position

        # End of inner while loop (temp_mt5_idx)

        if not match_found:
            # This 'else' executes if the inner loop finished *without* break
            # (i.e., ran out of MT5 tokens while trying to complete a Type 1 partial match)
            # log.warning(
            #     f"Ran out of MT5 subtokens while trying to match CKIP token '{ckip_token}' (index {ckip_idx}). "
            #     f"Partial reconstruction: '{current_reconstruction}'. "
            #     f"Padding involved MT5 subtokens: {subtoken_indices_in_match_attempt}"
            # )
            # Pad the subtokens we processed for this incomplete match
            for idx in subtoken_indices_in_match_attempt:
                 if aligned_tags[idx] == pad_tag_id:
                    aligned_tags[idx] = pad_tag_id
            # No more MT5 tokens, break the outer loop as well
            break

    # Log remaining unaligned tokens (optional)
    remaining_ckip = len(ckip_tokens) - ckip_idx
    # if remaining_ckip > 0:
    #     log.warning(f"{remaining_ckip} CKIP tokens remained unaligned at the end.")
    # Any remaining MT5 tokens at the end will keep their default pad_tag_id

    # log.debug(f"Alignment finished. Final aligned_tags: {aligned_tags}")
    return aligned_tags
def align_tags_to_subtokens2(
    ckip_tokens: list[str],
    ckip_tags: list[int],
    mt5_input_ids: list[int],
    tokenizer: PreTrainedTokenizerFast,
    pad_tag_id: int = -100 # Default in func signature, WILL BE OVERRIDDEN IN CALLS
) -> list[int]:
    """
    Aligns word-level tags (from CKIP) to subword tokens (from MT5).
    Handles cases where MT5 tokenization is finer (e.g., '翻譯' -> ' 翻', '譯')
    and coarser (e.g., '很', '好' -> ' 很好') than CKIP.
    Assigns the original tag to the first subtoken corresponding to the start
    of a CKIP word. Subsequent subtokens within the same original word (in the
    finer case) get pad_tag_id. When MT5 is coarser, the single MT5 subtoken
    gets the tag of the *first* CKIP token it represents.
    Args:
        ckip_tokens: List of word tokens from CKIP.
        ckip_tags: List of tags corresponding to ckip_tokens.
        mt5_input_ids: List of input IDs from the MT5 tokenizer for the *same* text
                       (potentially including start/end special tokens like </s>).
        tokenizer: The (fast) Hugging Face tokenizer instance (e.g., MT5TokenizerFast).
        pad_tag_id: ID for padding subtokens or unaligned tokens.
    Returns:
        A list of tag IDs aligned with mt5_input_ids.
    """
    if len(ckip_tokens) != len(ckip_tags):
        # Use logging or warning instead of raising error for robustness in mapping
        #log.error(f"Length mismatch: CKIP tokens ({len(ckip_tokens)}) vs tags ({len(ckip_tags)}). Returning padding.")
        # Return padding for the length of mt5_input_ids as a fallback
        return [pad_tag_id] * len(mt5_input_ids)

    # It's better to align based on tokens/offsets if possible, but using decoded tokens is common
    mt5_subtokens = tokenizer.convert_ids_to_tokens(mt5_input_ids)
    aligned_tags = [pad_tag_id] * len(mt5_input_ids) # Initialize based on MT5 length

    # Don't log excessively during normal runs unless debugging
    # log.debug(f"Starting alignment: CKIP Tokens={ckip_tokens}, MT5 Subtokens={mt5_subtokens}")

    ckip_idx = 0
    mt5_idx = 1 # Start MT5 index from 0

    while ckip_idx < len(ckip_tokens) and mt5_idx < len(mt5_subtokens):
        ckip_token = ckip_tokens[ckip_idx]
        ckip_tag = ckip_tags[ckip_idx]
        # log.debug(f"\nProcessing CKIP Token #{ckip_idx}: '{ckip_token}' (Tag: {ckip_tag}) | Starting at MT5 Index {mt5_idx}")

        # Skip empty CKIP tokens
        if not ckip_token:
             # log.debug(f"  Skipping empty CKIP token at index {ckip_idx}")
             ckip_idx += 1
             continue

        current_reconstruction = ""
        first_subtoken_idx_for_word = -1
        subtoken_indices_in_match_attempt = []
        temp_mt5_idx = mt5_idx
        match_found = False

        while temp_mt5_idx < len(mt5_subtokens):
            subtoken = mt5_subtokens[temp_mt5_idx]
            # SentencePiece cleaning
            cleaned_subtoken = subtoken.replace(' ', '') if subtoken.startswith(' ') else subtoken
            # Handle case where subtoken is just ' ' -> becomes empty after cleaning
            if not cleaned_subtoken and subtoken == ' ': cleaned_subtoken = ''

            # log.debug(f"  Considering MT5 Subtoken #{temp_mt5_idx}: '{subtoken}' | Cleaned: '{cleaned_subtoken}'")

            subtoken_indices_in_match_attempt.append(temp_mt5_idx)

            # Track first non-space subtoken index for the current CKIP word attempt
            if first_subtoken_idx_for_word == -1 and subtoken != ' ':
                 first_subtoken_idx_for_word = temp_mt5_idx

            # Build potential reconstruction using cleaned subtokens
            potential_reconstruction = current_reconstruction + cleaned_subtoken
            # log.debug(f"    Potential Reconstruction: '{potential_reconstruction}'")

            # --- Alignment Check 1: CKIP token starts with reconstruction ---
            if ckip_token.startswith(potential_reconstruction):
                # log.debug(f"    Match Type 1: CKIP '{ckip_token}' starts with Rec '{potential_reconstruction}'")
                current_reconstruction = potential_reconstruction

                # Perfect match found
                if current_reconstruction == ckip_token:
                    # log.debug(f"    >>> Full Match (Type 1) Found for '{ckip_token}' <<<")
                    if first_subtoken_idx_for_word != -1:
                        # log.debug(f"    Assigning tag {ckip_tag} to index {first_subtoken_idx_for_word}")
                        aligned_tags[first_subtoken_idx_for_word] = ckip_tag
                        # Pad subsequent subtokens within this match (excluding the first one)
                        for idx in subtoken_indices_in_match_attempt:
                            if idx != first_subtoken_idx_for_word:
                                # log.debug(f"    Assigning pad tag {pad_tag_id} to index {idx}")
                                aligned_tags[idx] = pad_tag_id
                    else: # Should not happen if match occurred, but safety
                         #log.warning(f"Matched '{ckip_token}' but no valid first subtoken index found. Padding {subtoken_indices_in_match_attempt}")
                         for idx in subtoken_indices_in_match_attempt: aligned_tags[idx] = pad_tag_id

                    ckip_idx += 1
                    mt5_idx = temp_mt5_idx + 1 # Consume MT5 tokens up to here
                    match_found = True
                    # log.debug(f"    Advancing CKIP to {ckip_idx}, MT5 to {mt5_idx}. Aligned: {aligned_tags}")
                    break # Exit inner loop for this CKIP token

                # Else: Partial match, continue consuming MT5 tokens
                temp_mt5_idx += 1
                continue

            # --- Alignment Check 2: Reconstruction starts with CKIP token (MT5 coarser) ---
            # This check is crucial when the *first* subtoken considered for a CKIP word
            # already contains the whole CKIP word (e.g., MT5:' 很好', CKIP:'很').
            elif len(subtoken_indices_in_match_attempt) == 1 and \
                 potential_reconstruction.startswith(ckip_token):
                 # log.debug(f"    Match Type 2: Rec '{potential_reconstruction}' starts with CKIP '{ckip_token}'")
                 if first_subtoken_idx_for_word != -1:
                     # log.debug(f"    Assigning tag {ckip_tag} to index {first_subtoken_idx_for_word}")
                     # Only assign if not already tagged by a previous CKIP word mapping to the same MT5 token
                     if aligned_tags[first_subtoken_idx_for_word] == pad_tag_id:
                         aligned_tags[first_subtoken_idx_for_word] = ckip_tag
                     # else: # Keep the tag from the first CKIP word that mapped here
                         # log.debug(f"    Index {first_subtoken_idx_for_word} already has tag {aligned_tags[first_subtoken_idx_for_word]}, keeping it.")
                 else: # Safety
                    #log.warning(f"Type 2 Match '{ckip_token}' but no first subtoken index. Padding {subtoken_indices_in_match_attempt}")
                    for idx in subtoken_indices_in_match_attempt: aligned_tags[idx] = pad_tag_id

                 # Only advance CKIP index. MT5 index remains the same for the next CKIP word.
                 ckip_idx += 1
                 # mt5_idx is NOT advanced here. The next CKIP token will start checking from the same mt5_idx.
                 match_found = True
                 # log.debug(f"    Advancing CKIP to {ckip_idx}. MT5 index remains {mt5_idx}. Aligned: {aligned_tags}")
                 break # Exit inner loop, proceed to next CKIP token

            # --- Mismatch ---
            else:
                # This means ckip_token does NOT start with potential_reconstruction,
                # AND potential_reconstruction does NOT start with ckip_token (or it wasn't the first subtoken).
                # log.warning( # Reduce noise by using warning only for persistent issues
                warnings.warn( # Use warnings instead of log.warning
                    f"Mismatch: Cannot align CKIP token '{ckip_token}' (index {ckip_idx}) "
                    f"with MT5 subtoken '{subtoken}' (index {temp_mt5_idx}). "
                    f"CKIP: '{ckip_token}', Potential Rec: '{potential_reconstruction}'. "
                    f"Strategy: Skip CKIP token, retry from same MT5 position ({mt5_idx})."
                )
                # Only pad the very first subtoken considered if mismatch happens immediately
                if len(subtoken_indices_in_match_attempt) == 1 and first_subtoken_idx_for_word != -1:
                     if aligned_tags[first_subtoken_idx_for_word] == pad_tag_id:
                         aligned_tags[first_subtoken_idx_for_word] = pad_tag_id

                ckip_idx += 1 # Skip this CKIP token
                # mt5_idx remains unchanged, retry next CKIP token from same MT5 position
                match_found = True # Mark as handled to prevent outer 'else'
                # log.debug(f"    Advancing CKIP to {ckip_idx}. MT5 index remains {mt5_idx}. Aligned: {aligned_tags}")
                break # Exit inner loop

        # End of inner while loop (temp_mt5_idx)

        if not match_found:
            # This means the inner loop finished because temp_mt5_idx reached the end
            # warnings.warn( # Reduce noise
            #log.info( # Use info or debug level
                #f"Ran out of MT5 subtokens while trying to match CKIP token '{ckip_token}' (index {ckip_idx}). "
                #f"Partial reconstruction: '{current_reconstruction}'. Padding remaining involved subtokens."
            #)
            for idx in subtoken_indices_in_match_attempt:
                 if aligned_tags[idx] == pad_tag_id:
                    aligned_tags[idx] = pad_tag_id
            break # Exit outer loop

    # After loops, pad any remaining MT5 tokens that weren't touched
    # (This shouldn't usually be necessary if loops complete correctly, but safe)
    # for idx in range(mt5_idx, len(mt5_subtokens)):
    #      if aligned_tags[idx] == pad_tag_id:
    #          aligned_tags[idx] = pad_tag_id

    # log.debug(f"Alignment finished. Final aligned_tags: {aligned_tags}")
    return aligned_tags
# --- End of align_tags_to_subtokens definition ---


# ===== Custom Model Definition (as before) =====
class CustomMT5Model(MT5ForConditionalGeneration):
    # ... (init and forward methods as before) ...
    def __init__(self, config, num_pos_tags=62, num_dep_tags=87): # Added num_dep_tags
        super().__init__(config)
        self.pos_embedding = torch.nn.Embedding(num_pos_tags, config.d_model)
        self.dep_embedding = torch.nn.Embedding(num_dep_tags, config.d_model) # Add dependency tag embedding
        self.head_position_embeddings = torch.nn.Embedding(MAX_INPUT_LENGTH, config.d_model) # Add head position embeddings

        torch.nn.init.xavier_uniform_(self.pos_embedding.weight)
        torch.nn.init.xavier_uniform_(self.dep_embedding.weight)
        torch.nn.init.xavier_uniform_(self.head_position_embeddings.weight)
        self.pos_embedding.weight.requires_grad = True
        self.dep_embedding.weight.requires_grad = True
        self.head_position_embeddings.weight.requires_grad = True

        self.projection = torch.nn.Linear(config.d_model * 2, config.d_model)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward( # ... rest of forward method ...
        self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, pos_ids=None, head_ids=None, dep_label_ids=None, **kwargs ):
        run_encoder = encoder_outputs is None
        if run_encoder and input_ids is not None:
            if inputs_embeds is not None: raise ValueError("Cannot provide both input_ids and inputs_embeds")
            token_embeds = self.get_input_embeddings()(input_ids)
            batch_size, seq_len = input_ids.shape
            if pos_ids is not None:
                if pos_ids.shape != (batch_size, seq_len): raise ValueError(f"pos_ids shape {pos_ids.shape} != input_ids shape {(batch_size, seq_len)}")
                pos_embeds = self.pos_embedding(pos_ids) * 0.2
                token_embeds = token_embeds + pos_embeds
            if head_ids is not None:
                 if head_ids.shape != (batch_size, seq_len): raise ValueError(f"head_ids shape {head_ids.shape} != input_ids shape {(batch_size, seq_len)}")
                 head_pos_embeds = self.head_position_embeddings(head_ids)
                 token_embeds = token_embeds + head_pos_embeds
            if dep_label_ids is not None:
                 if dep_label_ids.shape != (batch_size, seq_len): raise ValueError(f"dep_label_ids shape {dep_label_ids.shape} != input_ids shape {(batch_size, seq_len)}")
                 dep_label_embeds = self.dep_embedding(dep_label_ids)
                 token_embeds = token_embeds + dep_label_embeds
            inputs_embeds = token_embeds
            input_ids = None
        elif run_encoder and inputs_embeds is not None:
            input_ids = None
        kwargs.pop('num_items_in_batch', None)
        outputs = super().forward( input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs )
        return outputs

# ===== Evaluation Function (with Alignment) =====
def evaluation_function(model, val_dataset, tokenizer, device, max_input_length, max_target_length, task_prefix):
    """
    Evaluates the model on the validation dataset using ROUGE-L.
    Performs alignment of tags similar to the training preprocessing.

    Args:
        model: The trained CustomMT5Model.
        val_dataset: The original validation Dataset (before .map()).
        tokenizer: The tokenizer instance (must be Fast).
        device: The device to run on ('cuda' or 'cpu').
        max_input_length: Max sequence length for input.
        max_target_length: Max sequence length for generation.
        task_prefix: The prefix used during training (e.g., "translate Chinese to Gloss: ").
    """
    rouge = Rouge()
    predictions = []
    references = []

    model.to(device)
    model.eval()

    # --- Determine Padding IDs ---
    # Assuming 0 for POS/Head padding, get dep padding ID (needs map or default)
    # If dep_label_map isn't available here, default to 0.
    # For robustness, get it from tokenizer if possible, or default.
    pos_pad_id = 0
    head_pad_id = 0
    # Try getting pad ID from config if possible, else default
    dep_pad_id = getattr(model.config, 'dep_pad_id', 0) # Or pass dep_label_map["<pad>"] if available

    print("Running evaluation with alignment...")
    decoder_start_id = model.config.decoder_start_token_id if hasattr(model.config, 'decoder_start_token_id') and model.config.decoder_start_token_id is not None else tokenizer.pad_token_id
    print(f"Using decoder_start_token_id: {decoder_start_id} ({tokenizer.decode(decoder_start_id)}) for generation.")

    prefix_tokens = tokenizer(task_prefix, add_special_tokens=False)["input_ids"]
    prefix_len = len(prefix_tokens)

    for example in tqdm(val_dataset):
        # --- Get original data ---
        input_text = example["input_text"] # Original text
        target_text = example["target_text"]
        # ---> Use correct keys from original dataset <---
        ckip_tokens = example["ckip_tokens"]
        ckip_pos_ids = example["pos_ids"]
        ckip_head_ids = example["head_ids_orig"]
        ckip_dep_label_ids = example["dep_label_ids_orig"]
        # ---> End key correction <---

        # --- Tokenize input WITH prefix ---
        input_with_prefix = task_prefix + input_text
        inputs = tokenizer(
            input_with_prefix,
            max_length=max_input_length,
            padding="max_length", # Pad to max_length for model input
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # --- Align tags ---
        # Get input_ids for alignment (unpadded, excluding prefix)
        # Need to handle truncation carefully if it happened
        unpadded_ids_with_prefix = tokenizer(input_with_prefix, truncation=True, max_length=max_input_length)["input_ids"]
        unpadded_ids_for_alignment = unpadded_ids_with_prefix[prefix_len:]

        if unpadded_ids_with_prefix[:prefix_len] != prefix_tokens:
             warnings.warn(f"Eval Prefix tokenization mismatch for example. Alignment might be inaccurate.")

        aligned_pos = align_tags_to_subtokens(ckip_tokens, ckip_pos_ids, unpadded_ids_for_alignment, tokenizer, pad_tag_id=pos_pad_id)
        aligned_head = align_tags_to_subtokens(ckip_tokens, ckip_head_ids, unpadded_ids_for_alignment, tokenizer, pad_tag_id=head_pad_id)
        aligned_dep = align_tags_to_subtokens(ckip_tokens, ckip_dep_label_ids, unpadded_ids_for_alignment, tokenizer, pad_tag_id=dep_pad_id)

        # --- Combine prefix padding and pad to MAX_INPUT_LENGTH ---
        num_mt5_tokens_total = len(unpadded_ids_with_prefix) # Length before padding
        padding_length = max_input_length - num_mt5_tokens_total

        prefix_padding_pos = [pos_pad_id] * prefix_len
        padded_aligned_pos = prefix_padding_pos + aligned_pos + [pos_pad_id] * padding_length
        padded_aligned_pos = padded_aligned_pos[:max_input_length]

        prefix_padding_head = [head_pad_id] * prefix_len
        padded_aligned_head = prefix_padding_head + aligned_head + [head_pad_id] * padding_length
        padded_aligned_head = padded_aligned_head[:max_input_length]
        padded_aligned_head = [min(max(h, 0), max_input_length - 1) for h in padded_aligned_head]


        prefix_padding_dep = [dep_pad_id] * prefix_len
        padded_aligned_dep = prefix_padding_dep + aligned_dep + [dep_pad_id] * padding_length
        padded_aligned_dep = padded_aligned_dep[:max_input_length]

        # --- Create Tensors for generation ---
        # Ensure shapes match the already padded inputs['input_ids']
        if len(padded_aligned_pos) != max_input_length or \
           len(padded_aligned_head) != max_input_length or \
           len(padded_aligned_dep) != max_input_length:
            warnings.warn(f"Eval: Final padded length mismatch for example! Forcing length {max_input_length}.")
            padded_aligned_pos = (padded_aligned_pos + [pos_pad_id]*max_input_length)[:max_input_length]
            padded_aligned_head = (padded_aligned_head + [head_pad_id]*max_input_length)[:max_input_length]
            padded_aligned_dep = (padded_aligned_dep + [dep_pad_id]*max_input_length)[:max_input_length]
            # Re-clamp head IDs just in case
            padded_aligned_head = [min(max(h, 0), max_input_length - 1) for h in padded_aligned_head]


        pos_ids_tensor = torch.tensor([padded_aligned_pos], dtype=torch.long).to(device)
        head_ids_tensor = torch.tensor([padded_aligned_head], dtype=torch.long).to(device)
        dep_label_ids_tensor = torch.tensor([padded_aligned_dep], dtype=torch.long).to(device)

        # --- Generate with aligned inputs ---
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"], # Padded by tokenizer
                attention_mask=inputs["attention_mask"], # From tokenizer
                pos_ids=pos_ids_tensor,       # Aligned & Padded
                head_ids=head_ids_tensor,     # Aligned & Padded
                dep_label_ids=dep_label_ids_tensor, # Aligned & Padded
                decoder_start_token_id=decoder_start_id,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode output
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(translated_text)
        references.append(target_text)

    # --- Calculate ROUGE ---
    # ... (rest of ROUGE calculation as before) ...
    if not predictions or not references:
        print("Warning: No predictions or references generated for ROUGE calculation.")
        return 0.0
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
        return scores['rouge-l']['f'],filtered_preds,filtered_refs
    except ValueError as e:
        print(f"Error calculating ROUGE: {e}")
        return 0.0

# ===== Data Collator Definition (as before) =====
class DataCollatorWithPos(DataCollatorForSeq2Seq):
    # ... (init and call methods using F.pad as before) ...
    def __init__(self, tokenizer, model=None, padding=True, max_length=None, pad_to_multiple_of=None, label_pad_token_id=-100, return_tensors="pt"):
        super().__init__(tokenizer, model=model, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, label_pad_token_id=label_pad_token_id, return_tensors=return_tensors)
        self.pos_pad_id = 0
        self.head_pad_id = 0
        self.dep_pad_id = 0

    def __call__(self, features):
        pos_id_tensors = [feature.pop("pos_ids") for feature in features]
        head_id_tensors = [feature.pop("head_ids") for feature in features]
        dep_label_id_tensors = [feature.pop("dep_label_ids") for feature in features]
        batch = super().__call__(features)
        max_length_in_batch = batch["input_ids"].shape[1]
        padded_pos_tensors = []
        for tensor in pos_id_tensors:
            padding_needed = max_length_in_batch - tensor.shape[0]
            padded_tensor = F.pad(tensor, (0, padding_needed), mode='constant', value=self.pos_pad_id)
            padded_pos_tensors.append(padded_tensor)
        padded_head_tensors = []
        for tensor in head_id_tensors:
            padding_needed = max_length_in_batch - tensor.shape[0]
            padded_tensor = F.pad(tensor, (0, padding_needed), mode='constant', value=self.head_pad_id)
            padded_head_tensors.append(padded_tensor)
        padded_dep_label_tensors = []
        for tensor in dep_label_id_tensors:
            padding_needed = max_length_in_batch - tensor.shape[0]
            padded_tensor = F.pad(tensor, (0, padding_needed), mode='constant', value=self.dep_pad_id)
            padded_dep_label_tensors.append(padded_tensor)
        batch["pos_ids"] = torch.stack(padded_pos_tensors)
        batch["head_ids"] = torch.stack(padded_head_tensors)
        batch["dep_label_ids"] = torch.stack(padded_dep_label_tensors)
        return batch
# --- END OF FILE CustomMT5Model_add_dep_pos.py ---