# Terjman-English-Morocco-Darija-Translation

Terjman is a Transformer model trained for the translation from English to Moroccan darija.
This repository provides the necessary and sufficient code used for training.
All models are finetuned on a the [English-to-Moroccan-Darija](BounharAbdelaziz/English-to-Moroccan-Darija) dataset, using a **A100-40GB** GPU. 

Model checkpoints are available on [Hugging Face](https://huggingface.co/atlasia) 🤗:

- [Terjman-Ultra](https://huggingface.co/atlasia/Terjman-Ultra)
- [Terjman-Large](https://huggingface.co/spaces/atlasia/Terjman-Large-v2)
- [Terjman-Nano](https://huggingface.co/atlasia/Terjman-Nano) 

Note: Currently developping the second version of Terjman, trained on a larger dataset with translation from many languages to moroccan darija and vice versa.

## Setup and Training

Start by installing the necessary dependencies:
```bash
pip install -r requirements.txt
```
Then run the training script for the version you would like to train:
```bash
python3 train.py
```
Note that the v2 code is probably going to change.

## Usage

Using our model for translation is simple and straightforward. 
You can integrate it into your projects or workflows via the Hugging Face Transformers library. 
Here's a basic example of how to use the model in Python:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("atlasia/Terjman-Ultra")
model = AutoModelForSeq2SeqLM.from_pretrained("atlasia/Terjman-Ultra")

# Define your Moroccan Darija Arabizi text
input_text = "Your english text goes here."

# Tokenize the input text
input_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Perform translation
output_tokens = model.generate(**input_tokens)

# Decode the output tokens
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("Translation:", output_text)
```

## Example

Bellow is an example of translating English input to Moroccan Darija in Arabic letters (Ary) using [Terjman-Ultra](https://huggingface.co/atlasia/Terjman-Ultra):

**Input**: "Hi my friend, can you tell me a joke in moroccan darija? I'd be happy to hear that from you!"

**Output**: "أهلا صاحبي، تقدر تقولي مزحة بالدارجة المغربية؟ غادي نكون فرحان باش نسمعها منك!"

## Training hyperparameters

The hyperparameters depends on the architecture and are summarized in the table bellow:

|                 | Training epochs | Batch size | Learning rate | weight decay | warmup ratio | Gradient accumulation steps | Gradient checkpointing |
|-----------------|-----------------|------------|---------------|--------------|--------------|-----------------------------|------------------------|
| Terjman-Supreme | 5               | 1          | 5e-4          | 0.01         | 0.03         | 4                           | True                   |
| Terjman-Ultra   | 25              | 4          | 2e-5          | 0.01         | 0.03         | 4                           | True                   |
| Terjman-Large   | 120             | 16         | 5e-4          | 0.01         | 0.03         | 4                           | True                   |
| Terjman-Nano    | 40              | 64         | 3e-5          | 0.01         | 0.03         | 4                           | True                   |


## Framework versions

- Transformers 4.40.2
- Pytorch 2.2.1+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1


## Feedback & Limitations

These model still has some limitations mainly due to the lack of data. More high quality data can help in the process. Would you have any feedback, suggestions, or encounter any issues, please don't hesitate to reach out :)