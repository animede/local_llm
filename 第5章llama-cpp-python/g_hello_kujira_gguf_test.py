from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
# LLMの準備
#tokenizer = AutoTokenizer.from_pretrained("DataPilot/ArrowPro-7B-KUJIRA")
llm = Llama(model_path="./models/DataPilot-ArrowPro-7B-KUJIRA-Q4_K_M.gguf",
               n_gpu_layers=35,
               n_ctx=2048
                 )
system="あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"

def build_prompt(sys_msg,user_query):
    #sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
    template = """[INST] <<SYS>>{}<</SYS>>{}[/INST]"""
    return template.format(sys_msg,user_query)

user="まどマギで一番かわいいキャラはだれ？"

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
print("final ans",ans)
