# Import necessary libraries
import wandb
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
from datasets import load_dataset
import argparse
from utils import (
    print_gpu_utilization, 
    count_total_tokens, 
    preprocess_function,
    preprocess_multilingual,
    compute_metrics,
)

from sacrebleu.metrics import BLEU, CHRF, TER


if __name__ == "__main__":
    
    # any-to-any translation model
    MULTILINGUAL = True # False, True
    
    # N.B: All trainings were on a A100-40Gb
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Trainer of a model that translate English text to Moroccan Darija.')
    parser.add_argument('--model_name', required=True, type=str, help='Model name: "3B", "1B", "Helsinki_240M", "Helsinki_77M_512", "Helsinki_77M_256"')
    parser.add_argument('--max_len', required=True, type=int, help='Maximum length of the input text')
    
    args = parser.parse_args()
    
    # Dataset to use
    # v2.0 and v2.1
    # DATA_PATH = "BounharAbdelaziz/Terjman-v2-English-Darija-Dataset-580K" # contains some bad samples, n_tokens \approx 78M
    # v2.2
    # DATA_PATH = "BounharAbdelaziz/Terjman-v2-English-Darija-Dataset-350K"   # filtered out most bad samples, n_tokens \approx 59M
    # v2.3
    DATA_PATH = "BounharAbdelaziz/Morocco-Darija-Translation-Dataset-22K-13-lang"   # kept only DODa (audio) and 10k samples from the 350K, n_tokens \approx 9M
    
    EVAL_DATA_PATH = "atlasia/TerjamaBench"
    
    
    # experiment versions:
    #   - 2.1: trained on 580K en-darija. source english, target darija, n_tokens \approx 78M
    #   - 2.2: trained on 350K en-darija. source english, target darija, n_tokens \approx 59M Large and nano finetuned from previous version
    #   - 2.3: trained on 22K samples for darija to 14 languages. source can be any language to any other, n_tokens \approx 9M. All models finetuned from version 2.2
    version = "2.3"
    
    # project name to appear in wandb
    project_name = f"Terjman-v{version}"
    
    # Load metrics
    metric_bleu = BLEU(effective_order=True)
    metric_chrf = CHRF()
    metric_ter = TER()
    
    # how to select best model
    METRIC_FOR_BEST_MODEL = 'bleu'
    
    # Training hyperparameters and arguments
    weight_decay=0.01
    max_grad_norm = 1.0
    save_total_limit=1
    predict_with_generate=True
    warmup_ratio=0.1
    gradient_checkpointing=True
    lr_scheduler_type = "linear"
    
    # Evaluation hyperparameters    
    eval_steps = 25
    save_steps = 25
    logging_steps = 5
    eval_strategy="steps"
    push_to_hub=False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len
    FP16_TRAINING = False
    BF16_TRAINING = True
    TRAIN_FROM_SCRATCH = False
    
    # for Nano and large, we start from previously trained models which already knows english-darija
    BASE_MODELS = { 
        "Helsinki_77M_256": "BounharAbdelaziz/Terjman-Nano-v2.2",                             # Terjman-Nano-MAX_LEN-256          BASE_MODEL = "Helsinki-NLP/opus-mt-en-ar"
        "Helsinki_77M_512": "BounharAbdelaziz/Terjman-Nano-v2.2",                             # Terjman-Nano-MAX_LEN-512          BASE_MODEL = "Helsinki-NLP/opus-mt-en-ar"
        "Helsinki_240M": "BounharAbdelaziz/Terjman-Large-v2.2",                               # Terjman-Large                     BASE_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-ar"
        "1B": "BounharAbdelaziz/Terjman-Supreme-v2.2",                                        # Terjman-Ultra
        "3B": "BounharAbdelaziz/Terjman-Supreme-v2.2"                                         # Terjman-Supreme
    }
    
    HUB_PATHS = {   
        "Helsinki_77M_256": f"BounharAbdelaziz/Terjman-Nano-v{version}-256",    # Terjman-Nano
        "Helsinki_77M_512": f"BounharAbdelaziz/Terjman-Nano-v{version}-512",    # Terjman-Nano
        "Helsinki_240M": f"BounharAbdelaziz/Terjman-Large-v{version}",          # Terjman-Large
        "1B": f"BounharAbdelaziz/Terjman-Ultra-v{version}",                     # Terjman-Ultra
        "3B": f"BounharAbdelaziz/Terjman-Supreme-v{version}"                    # Terjman-Supreme
    }
    
    BATCH_SIZES = { 
        "Helsinki_77M_256": 64,                                                 # Terjman-Nano-MAX_LEN-256 (80 also work fine)
        "Helsinki_77M_512": 64,                                                 # Terjman-Nano-MAX_LEN-512
        "Helsinki_240M": 16,                                                    # Terjman-Large
        "1B": 4,                                                                # Terjman-Ultra
        "3B": 1                                                                 # Terjman-Supreme
    }
    
    N_EPOCHS = {    
        "Helsinki_77M_256": 8,                                                  # Terjman-Nano-MAX_LEN-256
        "Helsinki_77M_512": 8,                                                  # Terjman-Nano-MAX_LEN-512
        "Helsinki_240M": 10, #6,                                                     # Terjman-Large was 120 then 100 in v2.0
        "1B": 3, #10, # 6,                                                                # Terjman-Ultra was 25, now training with 1 and 6. 30 in v2.0
        "3B": 2, #6,                                                                # Terjman-Supreme was 5
    }
    
    LERANING_RATES = {  
        "Helsinki_77M_256": 5e-3,                                               # Terjman-Nano-MAX_LEN-256
        "Helsinki_77M_512": 1e-4,                                               # Terjman-Nano-MAX_LEN-512
        "Helsinki_240M": 1e-3, #5e-3, # 1e-4,                                   # Terjman-Large was 5e-4 in v2.0
        "1B": 5e-5, #5e-3, #5e-4,                                                      # Terjman-Ultra 5e-4 in v2.0
        "3B": 5e-5, #5e-4,                                                          # Terjman-Supreme 5e-4 in v2.0
    }
    
    GRAD_ACC = {    
        "Helsinki_77M_256": 4,                                                  # Terjman-Nano-MAX_LEN-256
        "Helsinki_77M_512": 4,                                                  # Terjman-Nano-MAX_LEN-512
        "Helsinki_240M": 8,                                                     # Terjman-Large
        "1B": 32,                                                               # Terjman-Ultra
        "3B": 128                                                               # Terjman-Supreme
    }
    
    # Tokenizer if training from scratch
    MY_TOKENIZER_PATH = "BounharAbdelaziz/Moroccan-Darija-Tokenizer-80k"
        
    gradient_accumulation_steps=GRAD_ACC[MODEL_NAME]
    BASE_MODEL = BASE_MODELS[MODEL_NAME]
    HUB_PATH = HUB_PATHS[MODEL_NAME]
    batch_size = BATCH_SIZES[MODEL_NAME]
    n_epochs = N_EPOCHS[MODEL_NAME]
    learning_rate = LERANING_RATES[MODEL_NAME]
    
    use_flash_attention_2 = False #True if MODEL_NAME in ["3B"] else False
    
    fp16 = '-FP16' if FP16_TRAINING else ''
    multilingual_tag = "ML" if MULTILINGUAL else ""
    
    run_name = f'{BASE_MODEL.split("/")[-1]}-bs-{batch_size*gradient_accumulation_steps}-lr-{learning_rate}-ep-{n_epochs}-wp-{warmup_ratio}-gnm-{max_grad_norm}{fp16}-mx-{MAX_LEN}-v{version}-{multilingual_tag}'
    HUB_PATH = f'{HUB_PATH}-bs-{batch_size*gradient_accumulation_steps}-lr-{learning_rate}-ep-{n_epochs}-wp-{warmup_ratio}-gnm-{max_grad_norm}-mx-{MAX_LEN}-v{version}-{multilingual_tag}'
    
    assert '--' not in run_name, f"[WARN] Detected -- in run_name. This will cause a push_to_hub error! Found run_name={run_name} "
    assert len(run_name) < 96, f"[WARN] run_name too long, found len(run_name)={len(run_name)} > 96. This will cause a push_to_hub error! Consider squeezing it. Found run_name={run_name}"


    assert '--' not in HUB_PATH, f"[WARN] Detected -- in HUB_PATH. This will cause a push_to_hub error! Found HUB_PATH={HUB_PATH} "
    assert len(HUB_PATH) < 96, f"[WARN] HUB_PATH too long, found len(HUB_PATH)={len(HUB_PATH)} > 96. This will cause a push_to_hub error! Consider squeezing it. Found HUB_PATH={HUB_PATH}"

    
    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged, all runs will be under this project
        project=project_name,   
        # Group runs by model size
        group=BASE_MODEL,       
        # Unique run name
        name=run_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "num_train_epochs": n_epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "weight_decay": weight_decay,
            "dataset": DATA_PATH,
            "multilingual": MULTILINGUAL,
        }
    )
    
    # NLLB models
    if MODEL_NAME in ["1B", "3B"]:
        TRAIN_NLLB = True
    # Helsinki models
    elif MODEL_NAME in ["Helsinki_77M_256", "Helsinki_77M_512", "Helsinki_240M"]:
        TRAIN_NLLB = False
    else:
        raise NotImplementedError(f'Model {MODEL_NAME} is not implemented yet! Choose from ["1B", "3B", "Helsinki_240M", "Helsinki_77M_512", "Helsinki_77M_256"]')
    
    print(f'[INFO] MODEL_NAME: {MODEL_NAME},  MAX_LEN: {MAX_LEN}, TRAIN_NLLB: {TRAIN_NLLB}')
    
    # check MAX_LEN for Helsinki models
    if MODEL_NAME == "Helsinki_77M_256":
        assert MAX_LEN == 256, f'Helsinki_77M_256 requires MAX_LEN set to 256'
    if MODEL_NAME == "Helsinki_77M_512":
        assert MAX_LEN == 512, f'Helsinki_77M_512 requires MAX_LEN set to 512'
    
    # Load training dataset from Hugging Face datasets
    train_dataset = load_dataset(DATA_PATH, split='train')
    
    # Load eval dataset (TerjamaBench)from Hugging Face datasets
    test_dataset = load_dataset(EVAL_DATA_PATH, split='test')
    
    print(f'[INFO] Training dataset loaded successfully. train_dataset: {train_dataset}')
    print(f'[INFO] Eval dataset loaded successfully. test_dataset: {test_dataset}')
    
    print("=" * 80)
    print(f'[INFO] Will finetune the model {MODEL_NAME} for {n_epochs} epochs with a total batch size of {batch_size * gradient_accumulation_steps} on the dataset {DATA_PATH} of n_samples={len(train_dataset)} and save it into {HUB_PATH} ...')
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    print(f'[INFO] Tokenizer vocab size: {len(tokenizer)}')
    
    if TRAIN_FROM_SCRATCH:
        
        # Set special tokens
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
            
    # Compile the model
    # model = torch.compile(model)
        
    # Count total number of tokens in the dataset
    count_total_tokens(train_dataset, "ary_Arab")
    
    # Initialize a data collator object
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # tokenize the dataset
    
    if MULTILINGUAL:
        print("[INFO] Processing dataset for multilingual training (any-to-any)")
        # For multilingual training, randomly select source and target languages for each example
        train_dataset = train_dataset.map(
            lambda x: preprocess_multilingual(
                x, 
                tokenizer=tokenizer, 
                max_length=MAX_LEN
            ), 
            batched=True, 
            batch_size=64,
            remove_columns=train_dataset.column_names  # Remove original columns to save memory
        )
    else:
        # For English to Darija only, use fixed language pair
        print("[INFO] Processing dataset for (english, ary_Arab) training")
        
        lang_pair = ("english", "ary_Arab", "eng_Latn", "ary_Arab")
        train_dataset = train_dataset.map(
            lambda x: preprocess_multilingual(
                x, 
                tokenizer=tokenizer, 
                max_length=MAX_LEN, 
                lang_pairs=lang_pair
            ), 
            batched=True, 
            batch_size=64,
            remove_columns=train_dataset.column_names # Remove original columns to save memory
        )
        
    # TerjamaBench contains only english darija pairs
    test_dataset = test_dataset.map(
        lambda x : preprocess_function(
            x, 
            tokenizer=tokenizer, 
            max_length=MAX_LEN, 
            source_lang="English", 
            target_lang="Darija",
        ), 
        batched=True, 
        batch_size=64
    )
    
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
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        report_to="wandb",
        load_best_model_at_end=True,
        max_grad_norm = max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        remove_unused_columns=True,
    )    

    # Initialize a Sequence to Sequence Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = lambda eval_preds: compute_metrics(
            eval_preds, 
            tokenizer, 
            metric_bleu, 
            metric_chrf,
            metric_ter
        ),
    )
    
    
    print(f'[INFO] GPU utilization before train()...')
    print_gpu_utilization()
    
    print("Start training")
    trainer.train()
    
    # Push the trained model to the Hugging Face hub
    trainer.push_to_hub(HUB_PATH)
