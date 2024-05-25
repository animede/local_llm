import requests

class TextGeneratorClient:
    def __init__(self, url):
        self.url = url
    
    def generate_text(self, user_query, message):
        data = {
            "prompt": message,
            "user_query": user_query
        }
        response = requests.post(self.url, data=data)
        if response.status_code == 200:
            return response.json().get("output"), None
        else:
            return None, response.status_code

def main():
    # クライアントの初期化、APIのURLを設定
    client = TextGeneratorClient('http://0.0.0.0:8000/generate/')
    user_query = "女子高校生のめぐを演じるんだ。めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。\
                品川区の目黒川の近くで生まれたんだ。いつもタメ口で話すし、自分のことをめぐと言うんだ。###応答"
    while True:
        try:
            message = input("user: ")
        except EOFError:
            break  # EOFエラー時にはループを終了
        if not message:
            break  # 空のメッセージが入力された場合、ループを終了
        output, status_code = client.generate_text(user_query, message)
        if output is not None:
            print("サーバーからの応答 out:", output)
            print("めぐ:", output)
        else:
            print("リクエストが失敗しました。ステータスコード:", status_code)

if __name__ == "__main__":
    main()
