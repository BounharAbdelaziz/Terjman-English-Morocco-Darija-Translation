import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
# from accelerate import Accelerator
from datasets import load_dataset, Dataset
import evaluate
import argparse
import numpy as np
from utils import print_gpu_utilization, count_total_tokens, preprocess_function, postprocess_text


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
    
    # Dataset to use
    DATA_PATH = "BounharAbdelaziz/English-to-Moroccan-Darija"
    
    # Training hyperparameters and arguments
    
    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len
    TRAIN_NLLB = True if args.train_nllb == 1 else False
    FP16_TRAINING = False
    BF16_TRAINING = True # False for Helsinki models, True for NLLB models
    
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
        
    MODEL_PATHS = { "Helsinki_77M_256": "Helsinki-NLP/opus-mt-en-ar",                       # Terjman-Nano-MAX_LEN-256
                    "Helsinki_77M_512": "Helsinki-NLP/opus-mt-en-ar",                       # Terjman-Nano-MAX_LEN-512
                    "Helsinki_240M": "Helsinki-NLP/opus-mt-tc-big-en-ar",                   # Terjman-Large
                    "1B": "facebook/nllb-200-1.3B",                                         # Terjman-Ultra
                    "3B": "facebook/nllb-200-3.3B"                                          # Terjman-Supreme
                }
    
    HUB_PATHS = {   "Helsinki_77M_256": "atlasia/Terjman-Nano-MAX_LEN-256",        # Terjman-Nano
                    "Helsinki_77M_512": "atlasia/Terjman-Nano-MAX_LEN-512",        # Terjman-Nano
                    "Helsinki_240M": "atlasia/Terjman-Large-v2",                   # Terjman-Large
                    "1B": "atlasia/Terjman-Ultra",                                 # Terjman-Ultra
                    "3B": "atlasia/Terjman-Supreme"                                # Terjman-Supreme
                }
    
    BATCH_SIZES = { "Helsinki_77M_256": 72,     # Terjman-Nano-MAX_LEN-256 (80 also work fine)
                    "Helsinki_77M_512": 64,     # Terjman-Nano-MAX_LEN-512
                    "Helsinki_240M": 16,        # Terjman-Large
                    "1B": 4,                    # Terjman-Ultra
                    "3B": 1                     # Terjman-Supreme
    }
    
    N_EPOCHS = {    "Helsinki_77M_256": 20,     # Terjman-Nano-MAX_LEN-256
                    "Helsinki_77M_512": 40,     # Terjman-Nano-MAX_LEN-512
                    "Helsinki_240M": 120,       # Terjman-Large
                    "1B": 25,                   # Terjman-Ultra
                    "3B": 5                    # Terjman-Supreme
    }
    
    LERANING_RATES = {  "Helsinki_77M_256": 3e-5,     # Terjman-Nano-MAX_LEN-256
                        "Helsinki_77M_512": 3e-5,     # Terjman-Nano-MAX_LEN-512
                        "Helsinki_240M": 5e-4,        # Terjman-Large
                        "1B": 2e-5,                   # Terjman-Ultra
                        "3B": 5e-4                    # Terjman-Supreme
    }
    
    MODEL_PATH = MODEL_PATHS[MODEL_NAME]
    HUB_PATH = HUB_PATHS[MODEL_NAME]
    batch_size = BATCH_SIZES[MODEL_NAME]
    n_epochs = N_EPOCHS[MODEL_NAME]
    learning_rate = LERANING_RATES[MODEL_NAME]
    
    weight_decay=0.01
    save_total_limit=3
    predict_with_generate=True
    warmup_ratio=0.03
    gradient_checkpointing=True
    gradient_accumulation_steps=4
    test_size = 0.01 # only 1% for testing as we do not have so many training data samples
    
    evaluation_strategy="epoch"
    push_to_hub=True
    
    source_lang="english"
    target_lang="darija_ar"
    metric = evaluate.load("sacrebleu")
    
    print(f'[INFO] GPU utilization before loading Model...')
    print_gpu_utilization()


    # Load training dataset from Hugging Face datasets
    dataset = load_dataset(DATA_PATH, split="train")
    
    # Safety check: Remove empty rows
    print(f'[INFO] Filter out empty examples...')
    df = dataset.to_pandas()
    df = df[['english', 'darija', 'includes_arabizi']]
    df = df.dropna(subset=[source_lang,target_lang])
    dataset = Dataset.from_pandas(df)
    print(f'[INFO] Filtering ended.')
        
    
    print("=" * 80)
    print(f'[INFO] Will finetune the model {MODEL_NAME} for {n_epochs} epochs with a batch size of {batch_size} on the dataset {DATA_PATH} and save it into {HUB_PATH} ...')
    print("=" * 80)

    # Load model directly
    if TRAIN_NLLB:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang="eng_Latn", tgt_lang="ar-SA")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if FP16_TRAINING:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    elif BF16_TRAINING:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    
    print(f'[INFO] GPU utilization after loading the model...')
    print_gpu_utilization()
    
    # Count total number of tokens in the dataset
    count_total_tokens(dataset, tokenizer, target_lang)
    
    # Split dataset into training and validation sets
    dataset = dataset.train_test_split(test_size=test_size)

    # Initialize a data collator object
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # tokenize the dataset
    tokenized_data = dataset.map(lambda x : preprocess_function(x, tokenizer=tokenizer, max_length=MAX_LEN, source_lang="english", target_lang="darija_ar"), batched=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=HUB_PATH,
        evaluation_strategy=evaluation_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=n_epochs,
        predict_with_generate=predict_with_generate,
        fp16=FP16_TRAINING,
        bf16=BF16_TRAINING,
        warmup_ratio=0.03,
        push_to_hub=push_to_hub,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=4,
        logging_dir="logs",
        report_to="tensorboard",
    )
    

    # Initialize a Sequence to Sequence Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    
    print(f'[INFO] GPU utilization before train()...')
    print_gpu_utilization()
    
    print("Start training")
    trainer.train()
    
    # Push the trained model to the Hugging Face hub
    trainer.push_to_hub(HUB_PATHS)
