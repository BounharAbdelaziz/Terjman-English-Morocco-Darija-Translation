import nest_asyncio
from tqdm import tqdm
from datasets import load_dataset
from googletrans import Translator
import asyncio

# Apply nest_asyncio to allow nested event loops in Jupyter Notebooks
nest_asyncio.apply()

async def translate_text(text, target_language):
    """Translates text asynchronously using googletrans."""
    translator = Translator()
    translations = await translator.translate(text, src='en', dest=target_language)
    return translations.text

def translate_batch(batch, target_language):
    """Processes a batch of texts and translates them."""
    texts = batch['english']
    translated_texts = asyncio.run(asyncio.gather(*[translate_text(text, target_language) for text in texts]))
    batch[target_language] = translated_texts
    return batch

def main():
    lang = "all"
    
    LANG_TRANS_DICT = {
        "arabic": "arb_Arab",
        "french": "fra_Latn",
        "german": "deu_Latn",
        "spanish": "spa_Latn",
        "portuguese": "por_Latn",
        "russian": "rus_Cyrl",
        "chinese_traditional": "zho_Hant",
        "japanese": "japanese",
        "korean": "korean",
        "greek": "ell_Grek",
        "italian": "italian",
        "persian": "persian",
        "turkish": "tur_Latn",
        "wolof": "wolof",
        "hindi": "hin_Deva",
        "hebrew": "heb_Hebr",
        # "sawahili": "sawahili",
    }
    
    BATCH_SIZE = 32  # Batch size for translating
    
    PUSH_TO_PATH = f"BounharAbdelaziz/Terjman-v2-English-Darija-Dataset-10K-extended-{lang}"
    DATA_PATH = "BounharAbdelaziz/Terjman-v2-English-Darija-Dataset-10K"
    
    # Load the dataset
    dataset = load_dataset(DATA_PATH, name="train")
    train_dataset = dataset['train'] #.select(range(BATCH_SIZE))  # Test with a small subset
    
    for language_name, iso_code in tqdm(LANG_TRANS_DICT.items(), desc="Translating all languages"):
        try:
            print(f"Translating to {language_name} ({iso_code})...")
            
            # Apply translation to the batch
            train_dataset = train_dataset.map(
                lambda examples: translate_batch(examples, language_name),  # Apply translation
                batched=True,  # Work on a batch of data
                batch_size=BATCH_SIZE,  # Define batch size
            )

        except Exception as e:
            print(f"Error translating {language_name}: {str(e)}")
        
        # Push the translated dataset to Hugging Face Hub
        train_dataset.push_to_hub(
            PUSH_TO_PATH, private=True, commit_message=f"Added {language_name} translations"
        )

if __name__ == "__main__":
    main()
