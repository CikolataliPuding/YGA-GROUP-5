from datasets import load_dataset

ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
print(ds.column_names)
print(ds[0])