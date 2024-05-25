from llama_cpp import Llama
import json
from pydantic import BaseModel
from fastapi import FastAPI,Form

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
               n_gpu_layers=35,
               n_ctx=2048,
               #tensor_split=[10,90],
               )

app = FastAPI()

class AnswerRequest(BaseModel):
     sys_msg : str
     user_query:str
     user:str
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
    user                  =gen_request.user
    talk_log_list=gen_request.talk_log_list
    log_f                =gen_request.log_f
    log_len           =gen_request.log_len
    max_token =gen_request.max_token
    top_k              =gen_request.top_k
    top_p              =gen_request.top_p
    get_temperature     =gen_request.temperature
    repeat_penalty         =gen_request.repeat_penalty
    frequency_penalty =gen_request.frequency_penalty
    print("top_k:",top_k,"top_p:",top_p,"get_temperature :",get_temperature ,"repeat_penalty:",repeat_penalty,"frequency_penalty:",frequency_penalty)

    talk_log_list= talk_log_list[0]
     
    prompt = sys_msg+"\n\n" + "### 指示: "+"\n" + user_query + "\n\n"  +  "### 入力:" +"\n"+ user + "\n\n"  +  "### 応答:"
    print("talk_log_list",talk_log_list)  
    #会話ヒストリ作成。プロンプトに追加する。
    log_len = int(log_len)
    if  log_f==True and log_len >0: # 履歴がTrueでログ数がゼロでなければtalk_log_listを作成
        sys_prompt=prompt.split("### 入力:")[0]
        talk_log_list.append( " \n\n"+ "### 入力:"+ " \n" + user+ " \n" )
        new_prompt=""
        for n in range(len(talk_log_list)):
            new_prompt=new_prompt + talk_log_list[n]
        prompt= sys_prompt + new_prompt+" \n \n"+ "### 応答:"+" \n"
    # 推論の実行
        """Sample a token from the model.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
        Returns:
            The sampled token.
              # デフォルトパラメータ
               top_k: int = 40,
               top_p: float = 0.95,
               temp: float = 0.80,
               repeat_penalty: float = 1.1,
               frequency_penalty: float = 0.0,
               presence_penalty: float = 0.0,
               tfs_z: float = 1.0,
               mirostat_mode: int = 0,
               mirostat_eta: float = 0.1,
               mirostat_tau: float = 5.0,
               penalize_nl: bool = True,
        """
    output = llm(
        prompt,
        stop=["### 入力","\n\n### 指示"],
        max_tokens=max_token,
        top_k = top_k ,
        top_p = top_p,
        temperature=get_temperature,
        repeat_penalty=repeat_penalty,
        frequency_penalty  =frequency_penalty,
        echo=True,
        )
    print(output["choices"][0])
    #output の"### 応答:"のあとに、"###"がない場合もあるので、ない場合は最初の"### 応答:"を選択
    try:
             ans = ans=output["choices"][0]["text"].split("### 応答:")[1].split("###")[0]
    except:
             ans = output["choices"][0]["text"].split("### 応答:")[1]
    print(ans)
    if len(talk_log_list)>log_len:
        talk_log_list=talk_log_list[2:] #ヒストリターンが指定回数を超えたら先頭(=一番古い）の会話（入力と応答）を削除
    talk_log_list.append("\n" +"###"+  "応答:"+"\n" + ans .replace("\n" ,""))
    result=200
    return {'message':result, "out":ans,"all_out":output,"prompt":prompt,"talk_log_list":talk_log_list }
     
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
