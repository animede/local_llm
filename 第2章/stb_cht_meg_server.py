from stb_cht_api_classg import TextGenerator
from fastapi import FastAPI,Form

app = FastAPI()

#初期化
text_generator = TextGenerator()

@app.post("/generate/")
def  genereate(prompt : str = Form(...), user_query: str =Form(...)):

     generated_text = text_generator.generate_response(prompt,user_query)
     print("system:", generated_text)
     return   {'output':  generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

