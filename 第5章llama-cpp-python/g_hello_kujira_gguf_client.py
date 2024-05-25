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
        self.sys_msg = "あなたは女子高校生のめぐです。\
めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。\
品川区の目黒川の近くで生まれたんだ。\
いつもタメ口で話すし、自分のことをめぐと言うんだ。回答には必ず日本語で答えるんだ。\
質問に対して意味のある返事をするんだ。"

    def generate_response(self, message):
        data = {
            "sys_msg": self.sys_msg,
            "user_query": message,
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

