import transformers
import torch
import time

class TextGenerator:
    def __init__(self, model_id="rinna/llama-3-youko-8b"):
        # モデルの初期化とロード
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    def generate_text(self, prompt):
        # テキスト生成メソッド
        if not prompt:
            print("exit...")
            return None
        
        start_time = time.time()
        output = self.pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True
        )
        generated_text = output[0]["generated_text"]
        generation_time = round(time.time() - start_time, 2)
        print("generation time=", generation_time, "秒")
        return generated_text

# メイン関数
def main():
    text_generator = TextGenerator()
    while True:
        try:
            message = input("user:")
        except EOFError:
            message = ""
        if not message:
            break
        generated_text = text_generator.generate_text(message)
        print("rinna:", generated_text)
        
if __name__ == "__main__":
    main()
