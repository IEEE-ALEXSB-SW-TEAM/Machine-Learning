from transformers import AutoTokenizer

models = ["bert-base-uncased", "gpt2", "t5-small", "roberta-base"]

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer("Hello Deep Learning!", return_tensors="pt")
    print(f"\nModel: {model}")
    print("Input IDs:", tokens["input_ids"][0].tolist())
    print("Tokens:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
