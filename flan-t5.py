from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
# from transformers import GenerationConfig

print("Loading dataset")

huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

print("Load FLAN-T5")

#model_name = 'google/flan-t5-base'
model_name = 'google/flan-t5-large'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def read_file_content(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

### 3.2 - Zero Shot Inference with the Prompt Template from FLAN-T5

print("Inference with prompting as of FLAN template")

#text = read_file_content("data/ac_2023v49_w927.txt")
text = read_file_content("data/polen_w460.txt")

prompt = f"Write a short summary for this text: {text}\n\nSummary:"


inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0],
    skip_special_tokens=True
)


print(f'INPUT PROMPT:\n{prompt}')
# print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')


if __name__ == '__main__':
    print("main")
