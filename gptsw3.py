import os
import sys
from datetime import datetime

from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

start_date = datetime.now().isoformat(timespec='hours')


def op_log(message="ping"):
    now = datetime.now().isoformat()
    print(now, " ", message)
    # Appending to a file
    with open(f'stdout_dump_{start_date}.txt', 'a', encoding="UTF-8") as file:
        file.write(f"{now} {message} \n")


def result_log(message="ping"):
    now = datetime.now().isoformat()
    print(now, " ", message)
    # Appending to a file
    with open(f'result_{start_date}.txt', 'a', encoding="UTF-8") as file:
        file.write(f"{now} {message} \n")


# Global token, nice .....
token = ""

gpt_sw3_instruct_XS = "AI-Sweden-Models/gpt-sw3-126m-instruct"
gpt_sw3_instruct_M = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
gpt_sw3_instruct_L = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct"
gpt_sw3_instruct_XL = "AI-Sweden-Models/gpt-sw3-20b-instruct"

gpt_sw3_base_XS = "AI-Sweden-Models/gpt-sw3-126m"
gpt_sw3_base_S = "AI-Sweden-Models/gpt-sw3-356m"
gpt_sw3_base_M = "AI-Sweden-Models/gpt-sw3-1.3b"
gpt_sw3_base_L = "AI-Sweden-Models/gpt-sw3-6.7b-v2"
gpt_sw3_base_XL = "AI-Sweden-Models/gpt-sw3-20b"
gpt_sw3_base_XXL = "AI-Sweden-Models/gpt-sw3-40b"


def load_tokenizer(model_name):
    op_log(f"Tokenizer: loading start {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    op_log(f"Tokenizer: loading finished {model_name}")
    return tokenizer


def question_and_answer(task_info, model, tokenizer, contextual_framework, task_query):
    prompt = chat_prompt(contextual_framework, task_query)

    op_log(f"Starting task/query {task_info} : {prompt}")
    start_time = datetime.now()
    op_log(f"Generating answer start {task_info}")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.6,
        top_p=1,
        eos_token_id=tokenizer.encode('<s>'),
        repetition_penalty=1.1
    )[0]
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    op_log(f"Generated answer finished {task_info}")
    stop_time = datetime.now()
    run_time = stop_time - start_time
    op_log(f"Task/query ({task_info}) [{len(prompt)}]: {prompt}")
    op_log(f"Model response: {response}")
    result_log(f"{task_info} [{len(prompt)}] [{len(response)}] {run_time}")
    op_log(f"Finished task/query {task_info}")
    return response


def chat_prompt(contextual_framework, task_query):
    prompt = f"""    
    <|endoftext|><s>
    User:
    {contextual_framework} 
    {task_query}
    <s>
    Bot:
    """.strip()
    return prompt


def limited(text, limit=200):
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def suffix_after(main_string: str, prefix: str) -> str:
    index = main_string.find(prefix)
    if index != -1:
        return main_string[index + len(prefix):]
    else:
        return ""


def count_words(s: str) -> int:
    return len(s.split())


def wc(s: str) -> str:
    return f"[c:{len(s)}, w:{count_words(s)}]"


def summary(task_info: str,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            text: str, summary_max_tokens=100) -> str:
    instruction = f"Sammanfatta följande text i en kort version på högst {summary_max_tokens} ord. Använd korta meningar."
    prompt = promt_summary(instruction, text)
    op_log(f"Starting task/query {task_info} : {limited(prompt)}")
    start_time = datetime.now()
    op_log(f"Generating response start {task_info} : {limited(prompt)}")
    input_ids = tokenizer.encode(prompt, max_length=len(prompt), truncation=True, return_tensors='pt')
    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=summary_max_tokens,
        do_sample=False,
        top_p=1,
        repetition_penalty=1.1
    )[0]
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=False).strip()
    op_log(f"Generated respomse finished {task_info}")
    stop_time = datetime.now()
    run_time = stop_time - start_time
    # op_log(f"Model response: {response}")
    answer = response_answer(response)
    op_log(f"Answer [{wc(answer)} {run_time}]: {answer}")
    op_log(f"{task_info} [Q:{len(prompt)}] [R:{len(response)}] [A:{len(answer)}] {run_time}")
    op_log(f"Finished task/query {task_info}")
    return answer


def response_answer(response):
    return suffix_after(response, "Bot:").strip()


def prompt_w_example(text):
    prompt_w_ex = (f"Sammanfatta följande artikel med högst 100 ord. "
                   f"Skriv i korta meningar med högst tio ord per mening. "
                   f"Använd följande som exempel på lämpligt format på sammanfattningen. \n"
                   f"Tjeckiens historia är rik och turbulent. Från slaviska stammar till Moraviska och Böhmiska riken under Přemyslid-dynastin. Blev intellektuell kraft i Heliga romerska riket. Efter trettioåriga kriget under Habsburg. Växande nationalism på 1800-talet. Tjeckoslovakien bildades 1918. Ockuperades av Nazityskland under andra världskriget. Sovjetkontroll efter kriget. Pragvåren 1968 krossades. Sammetsrevolutionen 1989 ledde till demokrati. Tjeckien blev självständig 1993, nu EU-medlem. Känd för kultur och historia.\n"
                   f"Här kommer artikeln du ska sammanfatta.:\n{text}")


def promt_wo_example(text):
    prompt = (
        f"Sammanfatta följande text om Polens historia i en kort version på högst 100 ord. Använd korta meningar med "
        f"maximalt tio ord per mening:\n"
        f"\n"
        f"{text}")
    return prompt


def promt_summary(instruction, text):
    return f"{endoftext_token()}{s_token()}User: {instruction}\n{text}{s_token()}Bot:"


def s_token():
    return "<s>"


def endoftext_token():
    return "<|endoftext|>"


def load_model(model_name: str) -> AutoModelForCausalLM:
    if os.path.exists("/mnt/gptsw3-models"):
        op_log(f"Model: loading from EBS start {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token, cache_dir="/mnt/gptsw3-models")
        op_log(f"Model: loading finished {model_name}")
    else:
        op_log(f"Model: loading start {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        op_log(f"Model configed using device {device}")
        model.eval()
        model.to(device)
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
        input_text = input("Du: ")
        # input_text = read_multiline_input("Du: ")
        if input_text.lower() == "exit":
            break
        question_and_answer(f"chat {name}", model, tokenizer,
                            "Du är lärare på lågstadiet. Svara som om det var ett barn som frågade. Men kort, inte mer än tio ord",
                            input_text)


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


def haiku_cold_luke_hot(model_name):
    op_log(f"Starting haikus {model_name}")
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    op_log(f"Cold haiku {model_name}")
    haiku(model, f"{model_name} cold", tokenizer)
    op_log(f"Luke haiku {model_name}")
    haiku(model, f"{model_name} luke", tokenizer)
    op_log(f"Hot haiku {model_name}")
    haiku(model, f"{model_name} hot", tokenizer)
    op_log(f"Finished haikus {model_name}")


def haiku(model, model_temperature, tokenizer):
    question_and_answer(f"haiku {model_temperature}", model, tokenizer, "Du är en poet som kan devops.",
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
    haiku_cold_luke_hot(gpt_sw3_base_XS)
    haiku_cold_luke_hot(gpt_sw3_instruct_XS)
    haiku_cold_luke_hot(gpt_sw3_base_S)
    haiku_cold_luke_hot(gpt_sw3_instruct_M)
    haiku_cold_luke_hot(gpt_sw3_base_M)
    haiku_cold_luke_hot(gpt_sw3_instruct_L)
    haiku_cold_luke_hot(gpt_sw3_base_L)


def summary_of_file(model_name, filename="polens_historia_wikipedia.txt", max_words=250):
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    content = read_file_content(filename)
    answer = summary(f"{model_name} summary of {filename}",
                     model, tokenizer,
                     content, summary_max_tokens=max_words)
    return answer


if __name__ == '__main__':
    op_log(f"Start of gptsw3.py on {os.environ.get('EC2_TYPE')} in region {os.environ.get('REGION')} ")
    command = sys.argv[1] if len(sys.argv) > 1 else "haikus"
    # long_running_task_with_periodic_updates(1200, 10)
    if command == "haiku":
        haiku_metrics()
    if command == "chat":
        chat_multiline_with_model(gpt_sw3_instruct_M)
    if command == "sammanfatta":
        file_name = sys.argv[2] if len(sys.argv) > 2 else "polens_historia_wikipedia.txt"
        max_words = int(sys.argv[3]) if len(sys.argv) > 2 else 100
        # summary_of_file(gpt_sw3_base_XS, file_name)
        # summary_of_file(gpt_sw3_base_S, file_name)
        # summary_of_file(gpt_sw3_base_M, file_name)
        answer = summary_of_file(gpt_sw3_instruct_M, file_name, max_words)
        op_log(f"Summary of {file_name} restricted to {max_words}: [{wc(answer)}:\n{answer}");
    # chat_multiline_with_model(sw3model)
    op_log("Shutdown")
