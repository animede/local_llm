import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained(
    "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
)
#以下を削除
#model = AutoModelForCausalLM.from_pretrained(
#    "stabilityai/japanese-stablelm-instruct-alpha-7b-v2",
#    trust_remote_code=True,
#    torch_dtype=torch.float16,
#    variant="fp16",
#)
#model.eval()
#if torch.cuda.is_available():
#    model = model.to("cuda")

#以下を追加
model = AutoModelForCausalLM.from_pretrained(
   "stabilityai/japanese-stablelm-instruct-alpha-7b-v2",
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",  
    variant="int8",
)
model.eval()



def build_prompt(user_query, inputs="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + user_query, ": \n"]
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

# Infer with prompt without any additional input
user_inputs = {
    "user_query": "与えられたことわざの意味を小学生でも分かるように教えてください。",
    "inputs": "情けは人のためならず"
}
prompt = build_prompt(**user_inputs)

input_ids = tokenizer.encode(
    prompt, 
    add_special_tokens=False, 
    return_tensors="pt"
)
#以下を追加
# パッドトークンIDの設定
pad_token_id = tokenizer.eos_token_id  # パディングトークンIDをeos_token_idに設定


tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=256,
    temperature=1,
    top_p=0.95,
    do_sample=True,
    pad_token_id= pad_token_id, #ここを追加
)

out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
print(out)
