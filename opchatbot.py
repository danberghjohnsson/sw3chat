from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


def doit():
    login("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")
    model = AutoModelForCausalLM.from_pretrained("AI-Sweden-Models/gpt-sw3-356m-instruct")
    tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-356m-instruct")
    prompt2 = "Han kom som ett yrväder en aprilafton och hade"
    while True:
        chat_in = input("Jaha ...:")

        prompt = f"""
    <|endoftext|><s>
    User:
    {chat_in}
    <s>
    Bot:
    """.strip()
        print(prompt)
        tokens = tokenizer.encode(chat_in, return_tensors='pt')
        print("generating")
        generated = model.generate(input_ids=tokens,
                                   max_new_tokens=100,
                                   repetition_penalty=1.1,
                                   do_sample=True,
                                   temperature=2.0)[0]
        print(generated)
        completion = tokenizer.decode(generated)
        print(completion)

if __name__ == '__main__':
    doit()