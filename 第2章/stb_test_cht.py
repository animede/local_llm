import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
import time

tokenizer = LlamaTokenizer.from_pretrained(
    "novelai/nerdstash-tokenizer-v1",
    additional_special_tokens=["▁▁"],
)
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

while True:
    try:
        message = input("user:")
    except EOFError:
        message = ""
    if not message:
        print("exit...")
        break
    start_time=time.time()
    user_inputs = {
        "user_query": "inputの質問に答えてください。",
        "inputs": message
    }
    prompt = build_prompt(**user_inputs)
    #print("prompt =",prompt )
          
    input_ids = tokenizer.encode(
        prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    )

    # パッドトークンIDの設定
    pad_token_id = tokenizer.eos_token_id  # パディングトークンIDをeos_token_idに設定

    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=256,
        temperature=1,
        top_p=0.95,
        do_sample=True,
        pad_token_id= pad_token_id, 
    )

    out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("genaraton time=",round(time.time()-start_time,2),"秒")
    out = out.split("###")[0]
    print(out)
