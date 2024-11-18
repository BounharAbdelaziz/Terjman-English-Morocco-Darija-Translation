import os
os.environ["TORCH_USE_CUDA_DSA"] = "1" # Enable CUDA DSA
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == "__main__":

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("atlasia/Terjman-Large-v2")
    model = AutoModelForSeq2SeqLM.from_pretrained("atlasia/Terjman-Large-v2")

    # Define your Moroccan Darija Arabizi text
    input_text = "Your english text goes here."

    # Tokenize the input text
    input_tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Perform translation
    output_tokens = model.generate(**input_tokens)

    # Decode the output tokens
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    print("Translation:", output_text)
    