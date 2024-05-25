import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("DataPilot/ArrowPro-7B-KUJIRA")
model = AutoModelForCausalLM.from_pretrained(
  "DataPilot/ArrowPro-7B-KUJIRA",
  torch_dtype="auto",
)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

def build_prompt(user_query):
    sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
    template = """[INST] <<SYS>>{}<</SYS>>{}[/INST]"""
    return template.format(sys_msg,user_query)

# Infer with prompt without any additional input
user_inputs = {
    "user_query": "まどマギで一番かわいいキャラはだれ？",
}
prompt = build_prompt(**user_inputs)
print(prompt)

input_ids = tokenizer.encode(
    prompt, 
    add_special_tokens=True, 
    return_tensors="pt"
)

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=500,
    temperature=1,
    top_p=0.95,
    do_sample=True,
)

out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
print(out)

