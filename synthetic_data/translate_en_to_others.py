from tqdm import tqdm
from datasets import load_dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def batch_translate(examples, batch_size, translator, MAX_LEN, target_language):
    """ Optimized batch translation function """
    # Get all English texts in current batch
    texts = examples['english']
    
    # Translate entire batch at once
    translations = translator(texts, src_lang="eng_Latn", tgt_lang=target_language, batch_size=batch_size, max_length=MAX_LEN)
    
    # print(f"text {texts[0]}")
    # print(f"translation {translations[0]['translation_text']}")
    
    # Extract just the translated text from results
    return {f'{target_language}': [t['translation_text'] for t in translations]}


if __name__ == "__main__":
    
    # Load the dataset
    dataset = load_dataset("BounharAbdelaziz/darija-translation-v5")

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

    MODEL_NAME = "facebook/nllb-200-1.3B"
    MAX_LEN = 512
    BATCH_SIZE = 8

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[INFO] device: ', device)
    print('[INFO] device_count: ', torch.cuda.device_count())

    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    translator = pipeline("translation", model=MODEL_NAME, tokenizer=tokenizer, max_length=MAX_LEN, device=device)

    train_dataset = dataset['train']

    for language_name, iso_code in tqdm(LANG_TRANS_DICT.items(), desc="Translating data"):
        print(f"Translating to {language_name} ({iso_code})...")
        # Apply translation with batching
        train_dataset[language_name] = train_dataset.map(
            lambda examples: batch_translate(examples, BATCH_SIZE, translator, MAX_LEN, target_language=iso_code),
            batched=True,  # Enable batching
            batch_size=BATCH_SIZE,
            desc="Translating"
        )
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': dataset['test'],
        })

        dataset_dict.push_to_hub("BounharAbdelaziz/darija-translation-v6", private=True, commit_message=f"Added translations to {language_name} using {MODEL_NAME}")