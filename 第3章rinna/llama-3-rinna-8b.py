import transformers
import torch

message="西田幾多郎は、"


model_id = "rinna/llama-3-youko-8b"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)
output = pipeline(
    message,
    max_new_tokens=256,
    do_sample=True
)
print(output)
