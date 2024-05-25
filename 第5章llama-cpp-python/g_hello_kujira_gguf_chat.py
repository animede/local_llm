from llama_cpp import Llama
import time
# LLMの準備
llm = Llama(model_path="./models/DataPilot-ArrowPro-7B-KUJIRA-Q4_K_M.gguf",
               n_gpu_layers=33,
               n_ctx=2048
                 )
system="あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"

def build_prompt(sys_msg,user_query):
    #sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
    template = """[INST] <<SYS>>{}<</SYS>>{}[/INST]"""
    return template.format(sys_msg,user_query)

while True:
    try:
        message = input("user: ")
    except EOFError:
        message = ""
    if not message:
        print("exit...")
        break
    start_time = time.time()
    user = message

    # Infer with prompt without any additional input
    user_inputs = {
          "sys_msg" :system,
          "user_query": user,
        }
    prompt = build_prompt(**user_inputs)
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.6,
        top_k=40,
        echo=False,
        )
    ans = output["choices"][0]["text"]
    print("genaraton time=",round(time.time()-start_time,2),"秒")
    print("final ans",ans)
