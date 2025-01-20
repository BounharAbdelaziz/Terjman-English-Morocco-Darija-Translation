from tqdm import tqdm
from datasets import load_dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from torch.cuda.amp import autocast
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class TranslationManager:
    def __init__(self, model_name, batch_size=16, max_length=512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()

    def setup_model(self):
        # Enable gradient checkpointing to reduce memory usage
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            # device_map="auto",  # Automatically handle multi-GPU setup
            # torch_dtype=torch.float16,  # Use half precision
            use_cache=False  # Disable KV cache for more memory efficiency
        )
        self.model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length
        )

    def batch_translate(self, examples, target_language):
        """Optimized batch translation with memory management"""
        texts = examples['english']
        
        with autocast():  # Enable automatic mixed precision
            translations = self.translator(
                texts,
                src_lang="eng_Latn",
                tgt_lang=target_language,
                batch_size=self.batch_size,
                max_length=self.max_length,
                num_beams=2  # Reduce beam search width for speed
            )
        
        return {target_language: [t['translation_text'] for t in translations]}

    def translate_language(self, dataset, language_name, iso_code):
        """Handle translation for a single language"""
        try:
            print(f"Translating to {language_name} ({iso_code})...")
            
            # Process in smaller chunks to manage memory better
            chunk_size = 1000
            num_chunks = len(dataset) // chunk_size + (1 if len(dataset) % chunk_size != 0 else 0)
            
            translated_chunks = []
            for i in tqdm(range(num_chunks), desc=f"Processing {language_name} chunks"):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(dataset))
                chunk = dataset.select(range(start_idx, end_idx))
                
                # Translate chunk
                translated_chunk = chunk.map(
                    lambda examples: self.batch_translate(examples, iso_code),
                    batched=True,
                    batch_size=self.batch_size,
                    remove_columns=chunk.column_names
                )
                
                translated_chunks.append(translated_chunk[iso_code])
                
                # Clear CUDA cache after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Combine all chunks
            dataset = dataset.add_column(language_name, sum(translated_chunks, []))
            return dataset
            
        except Exception as e:
            print(f"Error translating {language_name}: {str(e)}")
            return dataset

def main():
    LANG_TRANS_DICT = {
        "french": "fra_Latn",
        "german": "deu_Latn",
        "spanish": "spa_Latn",
        "arabic": "arb_Arab",
        "portuguese": "por_Latn",
        "russian": "rus_Cyrl",
        "chinese_traditional": "zho_Hant",
        "greek": "ell_Grek",
        "turkish": "tur_Latn",
        "hindi": "hin_Deva",
        "hebrew": "heb_Hebr",
    }

    # Configuration
    MODEL_NAME = "facebook/nllb-200-1.3B"
    BATCH_SIZE = 32  # Increased batch size
    MAX_LENGTH = 512
    
    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    
    # Initialize translation manager
    manager = TranslationManager(MODEL_NAME, BATCH_SIZE, MAX_LENGTH)
    
    # Load dataset
    dataset = load_dataset("BounharAbdelaziz/darija-translation-v5")
    train_dataset = dataset['train']
    
    # Process languages sequentially (more memory efficient than parallel)
    for language_name, iso_code in LANG_TRANS_DICT.items():
        train_dataset = manager.translate_language(train_dataset, language_name, iso_code)
        
        # Save progress after each language
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': dataset['test'],
        })
        
        dataset_dict.push_to_hub(
            "BounharAbdelaziz/darija-translation-v6",
            private=True,
            commit_message=f"Added translations to {language_name} using {MODEL_NAME}"
        )

if __name__ == "__main__":
    main()