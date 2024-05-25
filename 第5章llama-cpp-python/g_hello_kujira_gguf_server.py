from llama_cpp import Llama
import json
from pydantic import BaseModel
from fastapi import FastAPI,Form
import time

# LLMの準備
"""Load a llama.cpp model from `model_path`.
            model_path: Path to the model.
            seed: Random seed. -1 for random.
            n_ctx: Maximum context size.
            n_batch: Maximum number of prompt tokens to batch together when calling llama_eval.
            n_gpu_layers: Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
            main_gpu: Main GPU to use.
            tensor_split: Optional list of floats to split the model across multiple GPUs. If None, the model is not split.
            rope_freq_base: Base frequency for rope sampling.
            rope_freq_scale: Scale factor for rope sampling.
            low_vram: Use low VRAM mode.
            mul_mat_q: if true, use experimental mul_mat_q kernels
            f16_kv: Use half-precision for key/value cache.
            logits_all: Return logits for all tokens, not just the last token.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            embedding: Embedding mode only.
            n_threads: Number of threads to use. If None, the number of threads is automatically determined.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            numa: Enable NUMA support. (NOTE: The initial value of this parameter is used for the remainder of the program as this value is set in llama_backend_init)
            verbose: Print verbose output to stderr.
            kwargs: Unused keyword arguments (for additional backwards compatibility).
"""
llm = Llama(model_path="./models/japanese-stablelm-instruct-gamma-7b-q4_K_M.gguf",
               n_gpu_layers=33,
               n_ctx=2048,
               #tensor_split=[10,90],
               )

app = FastAPI()

def build_prompt(sys_msg,user_query):
    #sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
    template = """[INST] <<SYS>>{}<</SYS>>{}[/INST]"""
    return template.format(sys_msg,user_query)

class AnswerRequest(BaseModel):
     sys_msg : str
     user_query:str
     talk_log_list:list =[[]]
     log_f:bool = False
     log_len :int = 0
     max_token:int = 256
     temperature:float = 0.8
     repeat_penalty:float =  1.1
     top_k:int  = 40
     top_p:float = 0.95
     frequency_penalty:float = 0.0

@app.post("/generate/")
def  genereate(gen_request: AnswerRequest):
    sys_msg         =gen_request.sys_msg
    user_query  =gen_request.user_query
    talk_log_list=gen_request.talk_log_list
    log_f                =gen_request.log_f
    log_len           =gen_request.log_len
    max_token =gen_request.max_token
    top_k              =gen_request.top_k
    top_p              =gen_request.top_p
    get_temperature     =gen_request.temperature
    repeat_penalty         =gen_request.repeat_penalty
    frequency_penalty =gen_request.frequency_penalty

    # Infer with prompt without any additional input
    user_inputs = {
          "sys_msg" :sys_msg,
          "user_query": user_query,
        }
    prompt = build_prompt(**user_inputs)
    print("++++prompt=",prompt)
     
    # 推論の実行
    start_time=time.time()
    output = llm(
        prompt,
        stop=["[INST] ","/INST"],
        max_tokens=max_token,
        top_k = top_k ,
        top_p = top_p,
        temperature=get_temperature,
        repeat_penalty=repeat_penalty,
        frequency_penalty  =frequency_penalty,
        echo=False,
        )
    ans = output["choices"][0]["text"]
    print("genaraton time=",round(time.time()-start_time,2),"秒")
    print("final ans",ans)
    result=200
    return {'message':result, "out":ans,"prompt":prompt}
     
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
