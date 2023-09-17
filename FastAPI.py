from fastapi import FastAPI

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

model_path = r'EmailSubject'
model = load_model(model_path)
tokenizer = load_tokenizer(model_path)


def generate_text(sequence, max_new_tokens):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    input_length = ids.size(1)
    max_length = input_length + max_new_tokens
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)



@app.get("/subject/{prompt}")
async def root(prompt: str):
    print(prompt)
    return {"subject": generate_text("Email : " + prompt + " Subject : ", 7).split('Subject : ')[1]}

model_path = r'qamodel'
model1 = load_model(model_path)
tokenizer1 = load_tokenizer(model_path)

def generate_text1(sequence, max_new_tokens):
    ids = tokenizer1.encode(f'{sequence}', return_tensors='pt')
    input_length = ids.size(1)
    max_length = input_length + max_new_tokens
    final_outputs = model1.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

@app.get("/answer/{prompt}")
async def root(prompt: str):
    print(prompt)
    return {"Answer": generate_text1("Question: " + prompt + "Answer: ", 35).split('Answer: ')[1].split(".") + "."}