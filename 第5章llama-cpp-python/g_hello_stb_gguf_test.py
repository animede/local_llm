from llama_cpp import Llama
# LLMの準備
llm = Llama(model_path="./models/japanese-stablelm-instruct-gamma-7b-q4_K_M.gguf",n_ctx=512)
sys_msg="以下は、文脈のある入力と指示に基づいた会話です。入力の要求を適切に満たす応答を書きなさい。\
1回の入力に対して何度も応答してはいけません。語尾は「です。」や「ます。」や「ください。」は使ってはいけません。30文字で答えること。"
user_query="女子高校生のめぐを演じるんだ。\
めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。品川区の目黒川の近くで生まれたんだ。\
いつもタメ口で話すし、自分のことをめぐと言うんだ。"
user="名前を教えて。"
prompt =sys_msg+"\n\n" + "### 指示: "+"\n" + user_query + "\n\n"  +  "### 入力:" +"\n"+ user + "\n\n"  +  "### 応答:"
# 推論の実行
output = llm(
    prompt,
    max_tokens=256,
    temperature=0.6,
    top_k=40,
    stop=['### 指示'],
    echo=False,
    )
print("++++output=",output )
ans = output["choices"][0]["text"]
print("final ans",ans)

