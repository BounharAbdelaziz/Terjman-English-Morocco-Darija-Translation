import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gc
from tqdm import tqdm

def batch_translate(examples, batch_size, translator):
    """ Optimized batch translation function """
    # Get all English texts in current batch
    texts = examples['English']
    
    # Translate entire batch at once
    if 'Ultra' in MODEL_NAME or 'Supreme' in MODEL_NAME:
        translations = translator(texts, src_lang="eng_Latn", tgt_lang="ary_Arab", batch_size=batch_size, max_length=MAX_LEN)
    else:
        translations = translator(texts, batch_size=batch_size, max_length=MAX_LEN)
        
    # Extract just the translated text from results
    return {f'en2ary_{MODEL_NAME}_ary': [t['translation_text'] for t in translations]}


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("atlasia/TerjamaBench")
    
    # Model selection
    MODELS = {
        "BounharAbdelaziz/Terjman-Ultra-v2.0": 16,
        "BounharAbdelaziz/Terjman-Large-v2.0": 64,
        "BounharAbdelaziz/Terjman-Nano-v2.0-512": 64,
    }
     
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[INFO] device: ', device)
    print('[INFO] device_count: ', torch.cuda.device_count())
    
    # Sequence length
    MAX_LEN = 512
    
    # Loop through each model and run predictions using batching
    for MODEL_NAME, BATCH_SIZE in tqdm(MODELS.items(), desc="Processing Models"):
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f'[INFO] Running preds with Model: {MODEL_NAME}...')

        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        translator = pipeline("translation", model=MODEL_NAME, tokenizer=tokenizer, max_length=MAX_LEN, device=device)
        
        # Apply translation with batching
        dataset['test'] = dataset['test'].map(
            lambda examples: batch_translate(examples, BATCH_SIZE, translator),
            batched=True,  # Enable batching
            batch_size=BATCH_SIZE,
            desc="Translating"
        )
        
        print(f'[INFO] Finished translating with Model: {MODEL_NAME}...')
    
    print('[INFO] Done translating all models...')
    print('[INFO] Saving results...')
    
    dataset.push_to_hub("atlasia/Terjman_v2_on_TerjamaBench", commit_message="Added Terjaman Nano, Large and Ultra v2.0 translations")
    