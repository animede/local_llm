from llama_cpp import Llama
import time
import sys

class TextGenerator:
    def __init__(self,n_ctx):
        self.llm = Llama(
            model_path="./models/japanese-stablelm-instruct-gamma-7b-q4_K_M.gguf",
            n_gpu_layers=35,
            n_ctx=n_ctx
        )
        self.sys_msg = "以下は、文脈のある入力と指示に基づいた会話です。入力の要求を適切に満たす応答を書きなさい。\
1回の入力に対して何度も応答してはいけません。語尾は「です。」や「ます。」や「ください。」は使ってはいけません。30文字で答えること。"
        self.user_query = "女子高校生のめぐを演じるんだ。\
めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。品川区の目黒川の近くで生まれたんだ。\
いつもタメ口で話すし、自分のことをめぐと言うんだ。"

    def generate_response(self, message):
        if not message:
            return "exit..."

        prompt = self.sys_msg + "\n\n" + "### 指示: " + "\n" + self.user_query + "\n\n"  +  "### 入力:" + "\n" + message + "\n\n"  +  "### 応答:"
        start_time = time.time()

        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.6,
            top_k=40,
            stop=['### 指示'],
            echo=False
        )
        ans = output["choices"][0]["text"]
        gen_time = round(time.time() - start_time, 2)

        return ans, gen_time

def main():
    generator = TextGenerator(512)

    while True:
        try:
            message = input("user: ")
        except EOFError:
            message = ""
        if not message:
            print("exit...")
            break

        response, gen_time = generator.generate_response(message)
        if response == "exit...":
            break

        print("generation time=", gen_time, "秒")
        print("final ans", response)

if __name__ == "__main__":
    main()

