import os
import sys
from datetime import datetime
import time

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


def op_log(message="ping"):
    now = datetime.now().isoformat()
    print(now, " ", message)
    # Appending to a file
    with open('stdout_dump.txt', 'a') as file:
        now_message = now + " " + message + "\n"
        file.write(now_message)


# Global token, nice .....
token = ""

gpt_sw3_S = "AI-Sweden-Models/gpt-sw3-126m-instruct"
gpt_sw3_M = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
gpt_sw3_L = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct"
gpt_sw3_XL = "AI-Sweden-Models/gpt-sw3-20b-instruct"


def load_tokenizer(model_name):
    op_log(f"Tokenizer: loading start {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    op_log(f"Tokenizer: loading finished {model_name}")
    return tokenizer


def question_and_answer(model_name, model, tokenizer, contextual_framework, task_query):
    prompt = contextual_framework + " " + task_query
    op_log(f"Starting task/query {model_name} : {prompt}")
    start_time = datetime.now()
    op_log(f"Generating answer {model_name}")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=2048)[0]
    response = tokenizer.decode(output, skip_special_tokens=True)
    op_log(f"Generated answer {model_name}")
    stop_time = datetime.now()
    run_time = stop_time - start_time
    op_log(f"Task/query ({model_name}) [{len(prompt)}]: {prompt}")
    op_log(f"Model response [{len(response)}]: {response}")
    op_log(f"{model_name}: {run_time}")
    op_log(f"Finished task/query {model_name}")


def abbreviate(model_name, model, tokenizer):
    question_and_answer(model_name, model, tokenizer,
                        "Skapa en sammanfattning med de viktigaste punkterna från följande text i form av en punktlista",
                        polens_historia())


def load_model(model_name):
    if os.path.exists("/mnt/gptsw3-models"):
        op_log(f"Model: loading from EBS start {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token, cache_dir="/mnt/gptsw3-models")
        op_log(f"Model: loading finished {model_name}")
    else:
        op_log(f"Model: loading start {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
        op_log(f"Model: loading finished {model_name}")
    return model


def chat_multiline_with_model(name, answer_max_length=250):
    op_log("Tokenizer: loading " + name)
    tokenizer = load_tokenizer(name)
    op_log("Model: loading " + name)
    model = AutoModelForCausalLM.from_pretrained(name, token=token)
    op_log("Model: loaded " + name)
    print("Starting chat with GPT model. Type 'exit' to end.")
    while True:
        op_log("input")
        input_text = read_multiline_input("Du: ")

        if input_text.lower() == "exit":
            break
        prompt = f"Svara med max {answer_max_length} ord.\n {input_text}"
        op_log("Encoding the input text")
        # Encode the input text
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        op_log("Generating a response")
        # Generate a response
        output = model.generate(input_ids, max_length=answer_max_length)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(name, ":", response)


def read_multiline_input(prompt):
    print(prompt)
    lines = []
    finished = False
    while not finished:
        line = input()
        if line.strip() == "":
            finished = True
        else:
            lines.append(line)
    return "\n".join(lines)


def read_file_content(filename):
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"


def polens_historia():
    return read_file_content("polens_historia_wikipedia_mod.txt")


def haiku_cold_luke_hot(model_name):
    op_log(f"Starting haikus {model_name}")
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    op_log(f"Cold haiku {model_name}")
    haiku(model, model_name, tokenizer)
    op_log(f"Luke haiku {model_name}")
    haiku(model, model_name, tokenizer)
    op_log(f"Hot haiku {model_name}")
    haiku(model, model_name, tokenizer)
    op_log(f"Finished haikus {model_name}")


def haiku(model, model_name, tokenizer):
    question_and_answer(model_name, model, tokenizer, "Du är en poet som kan devops.",
                        "Skriv två haikus om kubernetes: en argumenterar för och en emot.")


def authenticate():
    global token
    op_log("authenticating")
    token = os.environ.get('HF_TOKEN')
    op_log(f"In python using: {token}")
    # Check if the token was found
    if not token:
        op_log("Error: Token not found in environment variables.")
        op_log("Please set the HF_TOKEN environment variable.")
        sys.exit(1)  # Exit the program with an error status
    login(token=token)
    op_log("authenticated")


def haiku_metrics():
    haiku_cold_luke_hot(gpt_sw3_S)
    haiku_cold_luke_hot(gpt_sw3_M)
    haiku_cold_luke_hot(gpt_sw3_L)
    haiku_cold_luke_hot(gpt_sw3_XL)


if __name__ == '__main__':
    op_log(f"Start of gptsw3.py on {os.environ.get('EC2_TYPE')} in region {os.environ.get('REGION')} ")
    # long_running_task_with_periodic_updates(1200, 10)
    authenticate()
    sw3model = gpt_sw3_S
    # model = load_model(model_name)
    # tokenizer = load_tokenizer(model_name)
    # abbreviate(model_name, model, tokenizer)
    haiku_metrics()
    # chat_multiline_with_model(sw3model)
    op_log("Shutdown")
