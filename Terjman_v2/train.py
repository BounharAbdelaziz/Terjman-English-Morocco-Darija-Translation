import torch
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
)
# from accelerate import Accelerator
from datasets import load_dataset, Dataset
import evaluate
import argparse
import numpy as np
from utils import print_gpu_utilization, count_total_tokens, preprocess_function, postprocess_text


# Function to filter out rows with missing values in source_lang or target_lang
def filter_missing_values(example, source_lang, target_lang):
    return example[source_lang] is not None and example[target_lang] is not None

def compute_metrics(eval_preds):
    
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == "__main__":
    
    # N.B: All trainings were on a A100-40Gb
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Trainer of a model that translate English text to Moroccan Darija.')
    parser.add_argument('--model_name', required=True, type=str, help='Model name: "3B", "1B", "Helsinki_240M", "Helsinki_77M_512", "Helsinki_77M_256"')
    parser.add_argument('--max_len', required=True, type=int, help='Maximum length of the input text')
    parser.add_argument('--train_nllb', required=True, type=int, help='Train NLLB model')
    
    args = parser.parse_args()
    
    TRAIN_FROM_SCRATCH = False
    
    BASE_MODELS = { "Helsinki_77M_256": "Helsinki-NLP/opus-mt-en-ar",                       # Terjman-Nano-MAX_LEN-256
                    "Helsinki_77M_512": "Helsinki-NLP/opus-mt-en-ar",                       # Terjman-Nano-MAX_LEN-512
                    "Helsinki_240M": "Helsinki-NLP/opus-mt-tc-big-en-ar",                   # Terjman-Large
                    "1B": "facebook/nllb-200-1.3B",                                         # Terjman-Ultra
                    "3B": "facebook/nllb-200-3.3B"                                          # Terjman-Supreme
                }
    
    HUB_PATHS = {   "Helsinki_77M_256": "atlasia/Terjman-Nano-v2.1-256",                    # Terjman-Nano
                    "Helsinki_77M_512": "atlasia/Terjman-Nano-v2.1-512",                    # Terjman-Nano
                    "Helsinki_240M": "atlasia/Terjman-Large-v2.1",                          # Terjman-Large
                    "1B": "atlasia/Terjman-Ultra-v2.1",                                     # Terjman-Ultra
                    "3B": "atlasia/Terjman-Supreme-v2.1"                                    # Terjman-Supreme
                }
    
    BATCH_SIZES = { "Helsinki_77M_256": 64,                                                 # Terjman-Nano-MAX_LEN-256 (80 also work fine)
                    "Helsinki_77M_512": 64,                                                 # Terjman-Nano-MAX_LEN-512
                    "Helsinki_240M": 32,                                                    # Terjman-Large
                    "1B": 8,                                                                # Terjman-Ultra
                    "3B": 1                                                                 # Terjman-Supreme
    }
    
    N_EPOCHS = {    "Helsinki_77M_256": 20,                                                 # Terjman-Nano-MAX_LEN-256
                    "Helsinki_77M_512": 10,                                                 # Terjman-Nano-MAX_LEN-512
                    "Helsinki_240M": 10,                                                    # Terjman-Large was 120 then 100 in v2.0
                    "1B": 5,                                                                # Terjman-Ultra was 25, now training with 1 and 6. 30 in v2.0
                    "3B": 2,                                                                # Terjman-Supreme was 5
    }
    
    LERANING_RATES = {  "Helsinki_77M_256": 3e-5,                                           # Terjman-Nano-MAX_LEN-256
                        "Helsinki_77M_512": 3e-5,                                           # Terjman-Nano-MAX_LEN-512
                        "Helsinki_240M": 5e-5,                                              # Terjman-Large was 5e-4 in v2.0
                        "1B": 1e-5,                                                         # Terjman-Ultra 5e-4 in v2.0
                        "3B": 1e-5,                                                         # Terjman-Supreme 5e-4 in v2.0
    }
    
    GRAD_ACC = {    "Helsinki_77M_256": 1,                                                  # Terjman-Nano-MAX_LEN-256
                    "Helsinki_77M_512": 1,                                                  # Terjman-Nano-MAX_LEN-512
                    "Helsinki_240M": 2,                                                     # Terjman-Large
                    "1B": 4,                                                                # Terjman-Ultra
                    "3B": 8                                                                 # Terjman-Supreme
    }
    
    # Dataset to use
    DATA_PATH = "BounharAbdelaziz/darija-translation-v5"
    
    TOKENIZER_PATH = "BounharAbdelaziz/Moroccan-Darija-Tokenizer-80k"
    
    # Training hyperparameters and arguments
    
    weight_decay=0.01
    save_total_limit=3
    predict_with_generate=True
    warmup_ratio=0.03
    gradient_checkpointing=True
    
    # Evaluation hyperparameters    
    eval_steps = 1000
    save_steps = 1000
    logging_steps = 100
    eval_strategy="epoch"
    push_to_hub=True
    
    source_lang="english"
    target_lang="darija_ar"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len
    TRAIN_NLLB = True if args.train_nllb == 1 else False
    FP16_TRAINING = False
    BF16_TRAINING = True # update in v2.1: now True for all. Was False for Helsinki models, True for NLLB models in v1.0
    
    gradient_accumulation_steps=GRAD_ACC[MODEL_NAME] # 4 for NLLB
    BASE_MODEL = BASE_MODELS[MODEL_NAME]
    HUB_PATH = HUB_PATHS[MODEL_NAME]
    batch_size = BATCH_SIZES[MODEL_NAME]
    n_epochs = N_EPOCHS[MODEL_NAME]
    learning_rate = LERANING_RATES[MODEL_NAME]
    
    use_flash_attention_2 = True if MODEL_NAME in ["3B"] else False
    
    
    print(f'[INFO] MODEL_NAME: {MODEL_NAME},  MAX_LEN: {MAX_LEN}, TRAIN_NLLB: {TRAIN_NLLB}')
        
    # NLLB models
    if MODEL_NAME in ["1B", "3B"]:
        assert TRAIN_NLLB == True, f'[ERROR] NLLB models requires TRAIN_NLLB set to True'
        assert MAX_LEN == 1024, f'[ERROR] NLLB models requires MAX_LEN set to 1024'
    
    # Helsinki models
    elif MODEL_NAME in ["Helsinki_77M_256", "Helsinki_77M_512", "Helsinki_240M"]:
        assert TRAIN_NLLB == False, f'[ERROR] Helsinki requires TRAIN_NLLB set to False'
    
    else:
        raise NotImplementedError(f'Model {MODEL_NAME} is not implemented yet! Choose from ["1B", "3B", "Helsinki_240M", "Helsinki_77M_512", "Helsinki_77M_256"]')
    
    # check MAX_LEN for Helsinki models
    if MODEL_NAME == "Helsinki_77M_256":
        assert MAX_LEN == 256, f'Helsinki_77M_256 requires MAX_LEN set to 256'
    if MODEL_NAME == "Helsinki_77M_512":
        assert MAX_LEN == 512, f'Helsinki_77M_512 requires MAX_LEN set to 512'
    if MODEL_NAME == "Helsinki_240M":
        assert MAX_LEN == 1024, f'Helsinki_240M requires MAX_LEN set to 1024'
    
    metric = evaluate.load("sacrebleu")
    
    # Load training dataset from Hugging Face datasets
    dataset = load_dataset(DATA_PATH)
    print(f'[INFO] Dataset loaded successfully.')
    print(dataset)
    
    # Safety check: Remove empty rows. Apply the filter to both train and test splits
    print(f'[INFO] Filter out empty examples...')
    dataset['train'] = dataset['train'].filter(lambda x: filter_missing_values(x, source_lang, target_lang))
    dataset['test'] = dataset['test'].filter(lambda x: filter_missing_values(x, source_lang, target_lang))
    print(f'[INFO] Filtering ended.')
            
    print("=" * 80)
    print(f'[INFO] Will finetune the model {MODEL_NAME} for {n_epochs} epochs with a batch size of {batch_size} on the dataset {DATA_PATH} of n_samples={len(dataset['train'])} and save it into {HUB_PATH} ...')
    print("=" * 80)

    # Load model directly
    if TRAIN_NLLB:
        if TRAIN_FROM_SCRATCH:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang="eng_Latn", tgt_lang="ary_Arab")
    else:
        if TRAIN_FROM_SCRATCH:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            
    # Set special tokens
    # tokenizer.pad_token = "<pad>"
    # tokenizer.unk_token = "<unk>"
    # tokenizer.bos_token = "<s>"
    # tokenizer.eos_token = "</s>"
    tokenizer.add_special_tokens({
                'pad_token': '<pad>', 
                'unk_token': '<unk>', 
                'bos_token': '<s>', 
                'eos_token': '</s>', 
                'mask_token': '[MASK]'
            })

    print(f'[INFO] tokenizer.pad_token_id: {tokenizer.pad_token_id}')
    print(f'[INFO] tokenizer.unk_token_id: {tokenizer.unk_token_id}')
    print(f'[INFO] tokenizer.bos_token_id: {tokenizer.bos_token_id}')
    print(f'[INFO] tokenizer.eos_token_id: {tokenizer.eos_token_id}')
    
    print(f'[INFO] Tokenizer vocab size: {len(tokenizer)}')
    
    if TRAIN_FROM_SCRATCH:
        
        # Configure model
        config = AutoConfig.from_pretrained(BASE_MODEL)
        config.vocab_size = len(tokenizer)
        config.decoder_vocab_size = len(tokenizer)
        config.max_position_embeddings = MAX_LEN
        config.pad_token_id = tokenizer.pad_token_id
        config.unk_token_id = tokenizer.unk_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.sep_token_id = tokenizer.sep_token_id
    
        if FP16_TRAINING:
            # Initialize model for sequence to sequence learning
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                config=config,
                torch_dtype=torch.float16,
                use_flash_attention_2=use_flash_attention_2,
                ignore_mismatched_sizes=True  # Ignore size mismatches when loading model from config
            ).to(device)
            
        elif BF16_TRAINING:
            # Initialize model for sequence to sequence learning
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                config=config,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=use_flash_attention_2,
                ignore_mismatched_sizes=True  # Ignore size mismatches when loading model from config
            ).to(device)
        else:
            # Initialize model for sequence to sequence learning
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                config=config,
                torch_dtype=torch.float32,
                use_flash_attention_2=use_flash_attention_2,
                ignore_mismatched_sizes=True  # Ignore size mismatches when loading model from config
            ).to(device)
            
        print(f"Model encoder vocabulary size: {config.vocab_size}")
        print(f"Model decoder vocabulary size: {config.decoder_vocab_size}")
    else:
        if FP16_TRAINING:
            # Initialize model for sequence to sequence learning
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                use_flash_attention_2=use_flash_attention_2,
            ).to(device)
            
        elif BF16_TRAINING:
            # Initialize model for sequence to sequence learning
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=use_flash_attention_2,
            ).to(device)
        else:
            # Initialize model for sequence to sequence learning
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                use_flash_attention_2=use_flash_attention_2,
            ).to(device)
        
    # Count total number of tokens in the dataset
    count_total_tokens(dataset['train'], tokenizer, target_lang)
    
    # Initialize a data collator object
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # tokenize the dataset
    train_dataset = dataset['train'].map(lambda x : preprocess_function(x, tokenizer=tokenizer, max_length=MAX_LEN, source_lang=source_lang, target_lang=target_lang), batched=True)
    test_dataset = dataset['test'].map(lambda x : preprocess_function(x, tokenizer=tokenizer, max_length=MAX_LEN, source_lang=source_lang, target_lang=target_lang), batched=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=HUB_PATH,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        num_train_epochs=n_epochs,
        predict_with_generate=predict_with_generate,
        fp16=FP16_TRAINING,
        bf16=BF16_TRAINING,
        warmup_ratio=warmup_ratio,
        push_to_hub=push_to_hub,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # logging_dir="logs",
        # report_to="tensorboard",
        load_best_model_at_end=True,
    )    

    # Initialize a Sequence to Sequence Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    
    print(f'[INFO] GPU utilization before train()...')
    print_gpu_utilization()
    
    print("Start training")
    trainer.train()
    
    # Push the trained model to the Hugging Face hub
    trainer.push_to_hub(HUB_PATH)
