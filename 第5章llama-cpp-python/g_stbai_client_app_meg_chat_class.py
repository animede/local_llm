import requests
import time

class Communicator:
    def __init__(self, url):
        self.url = url
    def send_request(self, data):
        # FastAPIサーバーにPOSTリクエストを送信
        response = requests.post(self.url, json=data)
        return response

class TextGenerator:
    def __init__(self, communicator):
        self.communicator = communicator
        self.sys_msg = "以下は、文脈のある入力と指示に基づいた会話です。入力の要求を適切に満たす応答を書きなさい。\
1回の入力に対して何度も応答してはいけません。語尾は「です。」や「ます。」や「ください。」は使ってはいけません。30文字で答えること。"
        self.user_query = "女子高校生のめぐを演じるんだ。\
めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。品川区の目黒川の近くで生まれたんだ。\
いつもタメ口で話すし、自分のことをめぐと言うんだ。"

    def generate_response(self, message):
        data = {
            "sys_msg": self.sys_msg,
            "user_query": self.user_query,
            "user": message,
            "max_token": 200,
            "temperature": 1,
        }
        response = self.communicator.send_request(data)
        
        if response.status_code == 200:
            result = response.json()
            output = result.get("out")
        else:
            print("リクエストが失敗しました。ステータスコード:", response.status_code)
        return output

def main():
    url = 'http://0.0.0.0:8000/generate/'  # FastAPIサーバーのURLに合わせて変更してください
    communicator = Communicator(url)
    generator = TextGenerator(communicator)

    while True:
        try:
            message = input("user: ")
        except EOFError:
            message = ""
        if not message:
            print("exit...")
            break
        start_time = time.time()
        output=generator.generate_response(message)
        print("genaraton time=",round(time.time()-start_time,2),"秒")
        print( output)

if __name__ == "__main__":
    main()

