import re
from pynvml import *

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
    
    inputs = [str(example) for example in examples[source_lang]]
    targets = [str(example) for example in examples[target_lang]]
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def postprocess_text(preds, labels):
    # Define a regex pattern to remove all punctuation
    remove_punct = re.compile(r'[^\w\s]')  # This keeps alphanumeric characters and spaces
    
    # Remove punctuation from predictions and labels using regex
    preds = [re.sub(remove_punct, '', pred).strip() for pred in preds]
    labels = [[re.sub(remove_punct, '', label[0]).strip()] for label in labels]
    
    # Remove extra spaces (i.e., multiple spaces between words)
    preds = [re.sub(r'\s+', ' ', pred) for pred in preds]
    labels = [[re.sub(r'\s+', ' ', label[0])] for label in labels]
    
    return preds, labels

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def count_total_tokens(dataset, target_lang="darija"):
    
    total_tokens = sum(dataset['darija_tokens'])
    print(f"[INFO] Total number of training tokens for column {target_lang} in the dataset: {total_tokens}")

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #