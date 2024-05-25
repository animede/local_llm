import requests
# メイン関数
def main():
    #TextGeneratorクラスの初期化、モデルのロード
    url='http://0.0.0.0:8000/generate/'
    user_query="女子高校生のめぐを演じるんだ。めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。\
                品川区の目黒川の近くで生まれたんだ。いつもタメ口で話すし、自分のことをめぐと言うんだ。###応答",
    while True:
        try:
            message = input("user:")
        except EOFError:
            message = ""
        if not message:
            break
        data ={"prompt":message,
               "user_query":user_query}
        response = requests.post(url, data=data)
        # レスポンスを表示
        if response.status_code == 200:
            result = response.json()
            print("サーバーからの応答out:", result.get("output"))
            generated_text = result.get("output")
        else:
            print("リクエストが失敗しました。ステータスコード:", response.status_code)   
        print("めぐ:", generated_text)
if __name__ == "__main__":
    main()
