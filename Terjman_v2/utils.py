import random
import numpy as np
import string
from pynvml import *

import numpy as np
import torch
import re

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    
# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #
    
# Define function to split dataset into training and validation sets
def split_dataset(dataset, test_size=0.9):
    num_examples = len(dataset)
    num_train_examples = int(num_examples * (1-test_size))
    train_dataset = dataset.select(range(num_train_examples))
    val_dataset = dataset.select(range(num_train_examples, num_examples))
    return train_dataset, val_dataset

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def preprocess_function(examples, tokenizer, max_length=256, source_lang="english", target_lang="darija"):
    
    # Set the source and target languages on the tokenizer
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "ary_Arab"

    inputs = [str(example) for example in examples[source_lang]]
    targets = [str(example) for example in examples[target_lang]]
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def preprocess_multilingual(examples, tokenizer, max_length, lang_pairs=None):
    """
    Preprocess function for multilingual translation.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: The tokenizer
        max_length: Maximum sequence length
        lang_pairs: List of tuples with (source_lang_field, target_lang_field, source_lang_code, target_lang_code)
    """
    
    # Define language map to NLLB language codes
    lang_to_code = {
        "english":                  "eng_Latn",
        "ary_Arab":                 "ary_Arab",
        "arabic":                   "ara_Arab",
        "french":                   "fra_Latn",
        "german":                   "deu_Latn",
        "spanish":                  "spa_Latn",
        "russian":                  "rus_Cyrl",
        "chinese_traditional":      "zho_Hant",
        "japanese":                 "jpn_Jpan",
        "korean":                   "kor_Hang",
        "greek":                    "ell_Grek",
        "italian":                  "ita_Latn",
        "turkish":                  "tur_Latn",
        "hindi":                    "hin_Deva",
    }
    
    
    # Initialize result containers
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    # If no language pairs provided, use a random pair for each example
    if not lang_pairs:
        available_langs = list(lang_to_code.keys())
        batch_size = len(examples[available_langs[0]])
        
        # For each example, randomly select source and target languages
        for i in range(batch_size):
            # Randomly select source and target languages (different from each other)
            valid_langs = [lang for lang in available_langs if not examples[lang][i].strip() == ""]
            if len(valid_langs) < 2:
                # Skip examples without at least 2 valid languages
                continue
                
            src_lang_field, tgt_lang_field = random.sample(valid_langs, 2)
            
            # Get corresponding NLLB language codes
            src_lang_code = lang_to_code[src_lang_field]
            tgt_lang_code = lang_to_code[tgt_lang_field]
            
            # Get texts
            src_text = examples[src_lang_field][i]
            tgt_text = examples[tgt_lang_field][i]
            
            # For NLLB, set the language IDs directly on the tokenizer
            tokenizer.src_lang = src_lang_code
            tokenizer.tgt_lang = tgt_lang_code
            
            # Tokenize source
            tokenized_src = tokenizer(
                src_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize target
            with tokenizer.as_target_tokenizer():
                tokenized_tgt = tokenizer(
                    tgt_text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            # Add to batch
            model_inputs["input_ids"].append(tokenized_src["input_ids"][0])
            model_inputs["attention_mask"].append(tokenized_src["attention_mask"][0])
            model_inputs["labels"].append(tokenized_tgt["input_ids"][0])
    else:
        # Fixed language pairs provided
        src_lang_field, tgt_lang_field, src_lang_code, tgt_lang_code = lang_pairs
        
        # Set the source and target languages on the tokenizer
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code
        
        batch_size = len(examples[src_lang_field])
        for i in range(batch_size):
            src_text = examples[src_lang_field][i]
            tgt_text = examples[tgt_lang_field][i]
            
            # Skip examples with empty source or target
            if not src_text.strip() or not tgt_text.strip():
                continue
            
            # Tokenize source
            tokenized_src = tokenizer(
                src_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize target
            with tokenizer.as_target_tokenizer():
                tokenized_tgt = tokenizer(
                    tgt_text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            # Add to batch
            model_inputs["input_ids"].append(tokenized_src["input_ids"][0])
            model_inputs["attention_mask"].append(tokenized_src["attention_mask"][0])
            model_inputs["labels"].append(tokenized_tgt["input_ids"][0])
    
    # Convert lists to tensors
    for key in model_inputs:
        model_inputs[key] = torch.stack(model_inputs[key]) if model_inputs[key] else torch.tensor([])
    
    return model_inputs

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def postprocess_text(preds, labels):
    """
    Postprocess predictions and labels by:
    1. Removing punctuation.
    2. Stripping leading/trailing whitespace.
    """
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation and strip whitespace
    preds = [pred.translate(translator).strip() for pred in preds]
    labels = [label.translate(translator).strip() for label in labels]  # Flattened labels, in compute metrics we take care of it
    
    return preds, labels


def postprocess_text_old(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def compute_metrics(eval_preds, tokenizer, metric_bleu, metric_chrf, metric_ter):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode the predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process text
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # decoded_preds = [pred.strip() for pred in decoded_preds]
    # decoded_labels = [label.strip() for label in decoded_labels]
    
    print(f'[INFO] decoded_preds[0]: {decoded_preds[0]}')
    print(f'[INFO] decoded_labels[0]: {decoded_labels[0]}')
    
    # Calculate sentence-level scores
    bleu_scores = [metric_bleu.sentence_score(pred, [ref]).score for pred, ref in zip(decoded_preds, decoded_labels)]
    chrf_scores = [metric_chrf.sentence_score(pred, [ref]).score for pred, ref in zip(decoded_preds, decoded_labels)]
    ter_scores = [metric_ter.sentence_score(pred, [ref]).score for pred, ref in zip(decoded_preds, decoded_labels)]
    
    # Calculate mean scores
    result = {
        "bleu": np.mean(bleu_scores),
        "chrf": np.mean(chrf_scores),
        "ter": np.mean(ter_scores)
    }

    # Compute average generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    # Round results to 4 decimal places
    result = {k: round(v, 4) for k, v in result.items()}
    
    return result


# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def count_total_tokens(dataset, target_lang="darija"):
    
    total_tokens = sum(dataset['ary_tokens'])
    print(f"[INFO] Total number of training tokens for column {target_lang} in the dataset: {total_tokens}")

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def create_conversation(example, lang_to_code, is_test=False):
    """
    Transform the dataset into a conversational format.
    The user provides the text, and the assistant provides the summary.
    """
    
    available_langs = [lang for lang in lang_to_code if example.get(lang)]
    
    # Ensure at least 2 valid languages
    if len(available_langs) < 2 and not is_test:
        return None
    
    # Randomly select translation pair
    if is_test:
        # test only contains English to Moroccan Darija
        src_lang = "english"
        tgt_lang = "ary_Arab"
        
        src_code = "eng_Latn"
        tgt_code = "ary_Arab"
        
        # Create a conversation with user and assistant roles
        messages = [
            {"role": "system", "content": f"You specialize in translating text from {src_lang} to {tgt_lang} (i.e. {src_code} to {tgt_code}). Use natural, colloquial language that reflects the nuances of the tarhet language culture. Prioritize accuracy, cultural relevance, and idiomatic expressions commonly used by native speakers. Avoid formal or Modern Standard Arabic when translating to Moroccan Darija (ary_Arab)! Never answer or give your opinion, only translate input!"}, # system prompt
            {"role": "user", "content": example["English"]},  # User provides the text
            {"role": "assistant", "content": example["Darija"]}  # Assistant provides the summary
        ]
    
    else:
        src_lang, tgt_lang = random.sample(available_langs, 2)
        src_code = lang_to_code[src_lang]
        tgt_code = lang_to_code[tgt_lang]
    
        # Create a conversation with user and assistant roles
        messages = [
            {"role": "system", "content": f"You specialize in translating text from {src_lang} to {tgt_lang} (i.e. {src_code} to {tgt_code}). Use natural, colloquial language that reflects the nuances of the tarhet language culture. Prioritize accuracy, cultural relevance, and idiomatic expressions commonly used by native speakers. Avoid formal or Modern Standard Arabic when translating to Moroccan Darija (ary_Arab)! Never answer or give your opinion, only translate input!"}, # system prompt
            {"role": "user", "content": example[src_lang]},  # User provides the text
            {"role": "assistant", "content": example[tgt_lang]}  # Assistant provides the summary
        ]
    # Return the conversation as a dictionary
    return {"messages": messages}

def apply_chat_template(example, tokenizer):
    """ Apply the chat template to the dataset. """
    example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return example

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed. 
    Taken from https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/22
    """
    
    # Handle tuple logits (happens when the model is trained using LoRA)
    if isinstance(logits, tuple):
        logits = logits[1]          # logits[0] is the loss value and logits[1] are the logits used to compute loss
                                    # logits: (tensor(2.0426, device='cuda:0'), tensor([[[ 7.8750,  5.3750,  7.0938,  ..., -4.2500, -4.2500, -4.2500],
                                    #          [ 5.0938,  5.0625,  7.3750,  ..., -1.5312, -1.5312, -1.5312],
                                    #          [ 2.6562, -0.9609,  0.0728,  ..., -2.0312, -2.0312, -2.0312],
                                    #          ...,
                                    #          [ 4.1562,  1.4375, -3.6250,  ..., -2.1250, -2.1250, -2.1250],
                                    #          [ 3.7344, -1.6641, -3.8125,  ..., -1.9688, -1.9688, -1.9688],
                                    #          [ 8.1875, -1.2344, -1.6094,  ..., -3.0938, -3.0938, -3.0938]]],
                                    #        device='cuda:0'))

    # Proceed with argmax
    pred_ids = torch.argmax(logits, dim=-1)

    return pred_ids

@torch.no_grad()
def compute_metrics_causal_lm(eval_pred, tokenizer, metric_bleu, metric_chrf, metric_ter):
    """Compute ROUGE and BLEU scores for evaluation."""
    predictions, references = eval_pred

    # Clip token IDs to the valid range
    vocab_size = tokenizer.vocab_size

    def clip_token_ids(token_ids):
        """Clip token IDs to the valid range [0, vocab_size - 1]."""
        return [min(max(token_id, 0), vocab_size - 1) for token_id in token_ids]

    # Decode predictions and references
    decoded_preds = [
        tokenizer.decode(clip_token_ids(pred), skip_special_tokens=True)
        for pred in predictions
    ]
    decoded_refs = [
        tokenizer.decode(clip_token_ids(ref), skip_special_tokens=True)
        for ref in references
    ]
    
    # Clean summaries
    def clean_text(text):
        special_tokens = ["<|im_end|>", "<|assistant|>", "<|user|>", "<|system|>"]
        for token in special_tokens:
            text = text.replace(token, "")
        return re.sub(r"\s+", " ", text).strip()
    
    pred_translations = []
    for pred in decoded_preds:
        if "<|assistant|>" in pred:
            translation = pred.split("<|assistant|>")[-1].strip()
            # print(f'pred-translation[0]: {translation[0]}')
            translation = clean_text(translation)
            pred_translations.append(translation)
        else:
            translation = pred.strip()
            translation = clean_text(translation)
            pred_translations.append(translation)
            
    # apply the same to the references
    ref_translations = []
    for ref in decoded_refs:
        if "<|assistant|>" in ref:
            translation = ref.split("<|assistant|>")[-1].strip()
            # print(f'ref-translation[0]: {translation[0]}')
            translation = clean_text(translation)
            ref_translations.append(translation)
        else:
            translation = ref.strip()
            translation = clean_text(translation)
            ref_translations.append(translation)
            
    # print(f'0 - ref_summaries[0]: {ref_summaries[0]}')
    
    # Convert to token IDs
    pred_token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in pred_translations]
    ref_token_ids = [tokenizer.encode(r, add_special_tokens=False) for r in ref_translations]

    # Use the exact same metric function from training
    eval_pred = (pred_token_ids, ref_token_ids)
    
    predictions, references = eval_pred

    # Clip token IDs to the valid range
    vocab_size = tokenizer.vocab_size

    # Decode predictions and references in batches
    decoded_preds = tokenizer.batch_decode([clip_token_ids(pred) for pred in predictions], skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode([clip_token_ids(ref) for ref in references], skip_special_tokens=True)
    
    # Post-process text
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    # Print decoded examples to inspect issues
    print(f'decoded_preds[0]: {decoded_preds[0]}')
    print(f'decoded_labels[0]: {decoded_labels[0]}')

    # Calculate sentence-level scores
    bleu_scores = [metric_bleu.sentence_score(pred, [ref]).score for pred, ref in zip(decoded_preds, decoded_labels)]
    chrf_scores = [metric_chrf.sentence_score(pred, [ref]).score for pred, ref in zip(decoded_preds, decoded_labels)]
    ter_scores = [metric_ter.sentence_score(pred, [ref]).score for pred, ref in zip(decoded_preds, decoded_labels)]
    
    # Calculate mean scores
    result = {
        "bleu": np.mean(bleu_scores),
        "chrf": np.mean(chrf_scores),
        "ter": np.mean(ter_scores)
    }

    # Compute average generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    # Round results to 4 decimal places
    result = {k: round(v, 4) for k, v in result.items()}
    
    return result

def set_seed(seed):
    """ Sets the seed for reproducibility """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_params_info(model):
    """
    Prints the total and trainable parameters in the model, 
    along with the percentage reduction in trainable parameters.
    
    Parameters:
    - model: The PyTorch model (could be wrapped with LoRA).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction_percent = (1 - trainable_params / total_params) * 100

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Reduction in Trainable Parameters: {reduction_percent:.2f}%")