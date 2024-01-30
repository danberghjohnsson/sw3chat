import os
import sys
from datetime import datetime

from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

instance_type = os.environ.get('EC2_TYPE')

start_date = datetime.now().isoformat(timespec='hours', sep='T')
dump_file = f'output/stdout_dump_{start_date}.txt'
result_file = f"output/result_{start_date}.txt"

gpt_sw3_instruct_XS = "AI-Sweden-Models/gpt-sw3-126m-instruct"
gpt_sw3_instruct_M = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
gpt_sw3_instruct_L = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct"
gpt_sw3_instruct_L_quant = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct-4bit-gptq"
gpt_sw3_instruct_XL = "AI-Sweden-Models/gpt-sw3-20b-instruct"

gpt_sw3_base_XS = "AI-Sweden-Models/gpt-sw3-126m"
gpt_sw3_base_S = "AI-Sweden-Models/gpt-sw3-356m"
gpt_sw3_base_M = "AI-Sweden-Models/gpt-sw3-1.3b"
gpt_sw3_base_L = "AI-Sweden-Models/gpt-sw3-6.7b-v2"
gpt_sw3_base_XL = "AI-Sweden-Models/gpt-sw3-20b"
gpt_sw3_base_XXL = "AI-Sweden-Models/gpt-sw3-40b"

def op_log(message="ping"):
    now = datetime.now().isoformat(sep='T')
    print(now, " ", message)
    # Appending to a file
    with open(dump_file, 'a', encoding="UTF-8") as file:
        file.write(f"{now} {message} \n")


def result_log(message="ping"):
    now = datetime.now().isoformat()
    print(now, " ", message)
    # Appending to a file
    with open(result_file, 'a', encoding="UTF-8") as file:
        file.write(f"{now} {message} \n")


# Global token, nice .....
token = ""






def load_tokenizer(model_name):
    op_log(f"Tokenizer: loading start {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    op_log(f"Tokenizer: loading finished {model_name}")
    return tokenizer


def question_and_answer(task_info, model, tokenizer, contextual_framework, task_query):
    prompt = chat_prompt(contextual_framework, task_query)

    op_log(f"Starting task/query {task_info} : {limited(prompt)}")
    start_time = datetime.now()
    op_log(f"Generating answer start {task_info}")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=2.0,
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
            text: str, summary_max_tokens, prompter) -> str:
    instruction = (f"Sammanfatta följande text i en kort version på högst {summary_max_tokens} ord. "
                   f"Använd korta meningar.")
    summary = prompter(instruction, text)
    prompt = summary
    op_log(f"Starting task/query {task_info} : {limited(prompt)}")
    start_time = datetime.now()
    op_log(f"Generating response start {task_info} : {limited(prompt)}")
    input_ids = tokenizer.encode(prompt, max_length=len(prompt), truncation=True, return_tensors='pt')
    op_log(f"Generating response, tokenized into tokens [{input_ids.size()}]: {input_ids}")
    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=summary_max_tokens,
        do_sample=False,
        top_p=1,
        repetition_penalty=1.1
    )[0]
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=False).strip()
    op_log(f"Generating response finished {task_info}")
    stop_time = datetime.now()
    run_time = stop_time - start_time
    # op_log(f"Model response: {response}")
    answer = response_answer(response)
    op_log(f"Answer [{wc(answer)} {run_time}]: {answer}")
    op_log(f"{task_info} [Q:{len(prompt)}] [R:{len(response)}] [A:{len(answer)}] {run_time}")
    op_log(f"Finished summary {task_info}")
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


def promt_summary_style(instruction, text):
    return f"{endoftext_token()}{s_token()}User: {instruction}\n{text}{s_token()}Bot:"

def promt_cv_match(cv, assignment):
    pre_text = ("Jag arbetar med en upphandling inom systemutveckling. Jag behöver avgöra hur väl en konsultprofil "
                "matchar uppdraget som det breskrivs i upphandlingen. Läs följande konsultprofil och beskrivning av uppdrag. "
                "Bedöm sedan konsultens lämplighet för uppdraget.")
    cv_text = (f"Konsultprofil:\n"
               f"{cv}")
    assignment_text = (f"Beskrivning av uppdraget:\n"
                       f"{assignment}")
    return (f"{endoftext_token()}{s_token()}"
            f"User: {pre_text}\n"
            f"{cv_text}\n"
            f"{assignment_text}\n"
            f"Beskriv hur väl konsultprofilen matchar det beskrivna uppdraget. \n"
            f"Ge även ett betyg enligt följande skala."
            f"5 (Perfekt): Konsultens färdigheter och erfarenhet matchar exakt med uppdragets krav."
            f"4 (Mycket bra): Konsulten har starka relevanta färdigheter, även om det inte är en exakt match."
            f"3 (Bra): Konsulten har relevanta färdigheter men kanske inte den fullständiga erfarenheten eller specialiseringen som krävs."
            f"2 (OK): Det finns viss överlappning i färdigheter och erfarenheter, men det är inte en idealisk match."
            f"1 (Dålig): Konsultens färdigheter och erfarenhetsområde överensstämmer inte med uppdragets krav."
            f"Svara med ett omdöme på högst 20 ord, sedan betyg mellan 1 och 5."            
            f"{s_token()}Bot:")


def s_token():
    return "<s>"


def endoftext_token():
    return "<|endoftext|>"


def load_model(model_name: str) -> PreTrainedModel:
    if os.path.exists("/mnt/gptsw3-models"):
        op_log(f"Model: loading from EBS start {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token, cache_dir="/mnt/gptsw3-models")
        op_log(f"Model: loading finished {model_name}")
    else:
        op_log(f"Model: loading start {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
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
    model = AutoModelForCausalLM.from_pretrained(name)
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


def summary_of_file(model_name, filename, max_words=250):
    content = read_file_content(filename)
    op_log(f"Making summary of {filename} {wc(content)} restricted to {max_words}")

    tokenizer = load_tokenizer(model_name)
    model: PreTrainedModel = load_model(model_name)

    answer = summary(f"{model_name} summary of {filename}",
                     model, tokenizer,
                     content, summary_max_tokens=max_words, prompter=promt_summary_style)
    result_log(f"Summary of {filename} {wc(content)} restricted to {max_words}: {wc(answer)}:\n{answer}")

def short_story_questions():
    model_name = gpt_sw3_instruct_L
    tokenizer = load_tokenizer(model_name)
    model: PreTrainedModel = load_model(model_name)
    story = read_file_content("data/novell_w1048.txt")
    with open("novell_fragor_ren.txt", 'r') as file:
        for story_question in file:
            op_log(story_question)
            instruction = (f"Läs följnade novell. Besvara sen frågan. \n"
                           f"{story}"
                           f"{story_question}")
            response = question_and_answer(f"Ansering {story_question}", model, tokenizer,
                                 "Du är litterärt kunning",
                                 instruction)
            result_log(f"{story_question}"
                       f"{response_answer(response)}")


def cv_match_assignment(tokenizer, model, assignment, cv):
    task_info = f"Matching {limited(cv, 20)} with {limited(assignment, 20)}"
    prompt =  promt_cv_match(cv, assignment)
    op_log(f"Starting task/query {task_info} : {limited(prompt)}")
    start_time = datetime.now()
    op_log(f"Generating response start {task_info} : {limited(prompt)}")
    input_ids = tokenizer.encode(prompt, max_length=len(prompt), truncation=True, return_tensors='pt')
    op_log(f"Generating response, tokenized into tokens [{input_ids.size()}]")
    generated_token_ids = model.generate(
        inputs=input_ids,
        max_new_tokens=120,
        do_sample=False,
        top_p=1,
        repetition_penalty=1.1
    )[0]
    response = tokenizer.decode(generated_token_ids, skip_special_tokens=False).strip()
    op_log(f"Generating response finished {task_info}")
    stop_time = datetime.now()
    run_time = stop_time - start_time
    # op_log(f"Model response: {response}")
    answer = response_answer(response)
    op_log(f"Answer [{wc(answer)} {run_time}]: {answer}")
    op_log(f"{task_info} [Q:{len(prompt)}] [R:{len(response)}] [A:{len(answer)}] {run_time}")
    op_log(f"Finished summary {task_info}")
    return answer


def cv_match_files(cv_file_name, assignment_file_name, model_name):
    op_log(f"Starting match of cv {cv_file_name} towards assignment {assignment_file_name}")
    cv = read_file_content(cv_file_name)
    assignment = read_file_content(assignment_file_name)
    op_log(f"Matching "
           f"cv {cv_file_name} {wc(cv)} {limited(cv, 50)} "
           f"with "
           f"assignment {assignment_file_name} {wc(assignment)} {limited(assignment, 50)} ")

    answer = cv_match_assignment(load_tokenizer(model_name),
                                 load_model(model_name),
                                 assignment,
                                 cv)
    result_log(f"Match of {model_name} "
               f"cv {limited(cv, 20)} {wc(cv)} "
               f"with "
               f"assignment  {limited(assignment, 20)} {wc(assignment)}:\n"
               f"{answer}")



if __name__ == '__main__':
    print(os.environ.get('HOSTNAME'))
    op_log(str(sys.argv))
    op_log(f"Start of gptsw3.py on {instance_type} in region {os.environ.get('REGION')} ")
    command = sys.argv[1] if len(sys.argv) > 1 else "haikus"
    start_date = datetime.now()
    # long_running_task_with_periodic_updates(1200, 10)
    if command == "novell":
        short_story_questions()
    if command == "haiku":
        haiku_metrics()
    if command == "chat":
        chat_multiline_with_model(gpt_sw3_instruct_M)
    if command == "analysera":
        file_name = sys.argv[2] if len(sys.argv) > 2 else "polens_historia_wikipedia.txt"
        max_words = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        instruct = sys.argv[2] if len(sys.argv) > 2 else f"Sammanfatta inom {max_words} ord"
        prompt_builder = lambda text : promt_summary_style("Detta har en VD skrivit. Hur går det för företaget?", text)
        summary_of_file(gpt_sw3_instruct_M, filename=file_name, max_words=max_words)
    if command == "sammanfatta":
        file_name = sys.argv[2] if len(sys.argv) > 2 else "polens_historia_wikipedia.txt"
        max_words = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        # summary_of_file(gpt_sw3_base_XS, file_name)
        # summary_of_file(gpt_sw3_base_S, file_name)
        # summary_of_file(gpt_sw3_base_M, file_name)
        summary_of_file(gpt_sw3_instruct_L, file_name, max_words)
    if command == "cv":
        cv_file_name = sys.argv[2]
        assignment_file_name = sys.argv[3]
        cv_match_files(cv_file_name, assignment_file_name, gpt_sw3_instruct_L)
    if command == "cv_all":
        for model in [gpt_sw3_instruct_L]:
            for assignment_n in range(1,5):
                assignment_file_name = f"data/cv_match/assignment_{assignment_n}.txt"
                for cv_n in range(1,6):
                    cv_file_name = f"data/cv_match/cv_{cv_n}.txt"
                    cv_match_files(cv_file_name, assignment_file_name, model)


    # chat_multiline_with_model(sw3model)
    stop_date = datetime.now()
    op_log(f"Shutdown after {stop_date-start_date}")
