import transformers
import torch
import time

model_id = "rinna/llama-3-youko-8b"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

while True:
    try:
        message = input("user:")
    except EOFError:
        message = ""
    if not message:
        print("exit...")
        break
    start_time=time.time()
    output = pipeline(
        message,
        max_new_tokens=256,
        do_sample=True
    )
    #print(output)
    ans=output[0]["generated_text"]
    print("rinna;",ans)
    print("genaraton time=",round(time.time()-start_time,2),"秒")
