import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM
import time

class TextGenerator:
    def __init__(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            additional_special_tokens=["▁▁"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/japanese-stablelm-instruct-alpha-7b-v2",
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto",  
            variant="int8",
        )
        self.model.eval()
    
    def build_prompt(self, user_query, inputs="", sep="\n\n### "):
        sys_msg = ("以下は、応答の指示に基づく、入力との会話です。入力の要求を適切に満たす応答を一回だけ書きなさい。\
                   1回の入力に対して何度も応答してはいけません。")
        p = sys_msg
        roles = ["指示", "応答"]
        msgs = [": \n" + user_query, ": \n"]
        if inputs:
            roles.insert(1, "入力")
            msgs.insert(1, ": \n" + inputs)
        for role, msg in zip(roles, msgs):
            p += sep + role + msg
        return p
    
    def generate_response(self, message):
        user_inputs = {
            "user_query": "女子高校生のめぐを演じるんだ。めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。\
                          品川区の目黒川の近くで生まれたんだ。いつもタメ口で話すし、自分のことをめぐと言うんだ。###応答",
            "inputs": message
        }
        prompt = self.build_prompt(**user_inputs)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        pad_token_id = self.tokenizer.eos_token_id
        tokens = self.model.generate(
            input_ids.to(device=self.model.device),
            max_new_tokens=30,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=pad_token_id,
        )

        out = self.tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        #print("org out=",out)
        out = out.split("###")[0].replace("\n", "")
        return out

def main():
    generator = TextGenerator()
    while True:
        try:
            message = input("user:")
        except EOFError:
            print("exit...")
            break
        if not message:
            print("exit...")
            break
        start_time = time.time()
        out = generator.generate_response(message)
        print("generation time=", round(time.time() - start_time, 2), "秒")
        print(out)

if __name__ == "__main__":
    main()

