import requests
import json
import time
# Infer with prompt without any additional input
data = {"sys_msg" : "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。",
    "user_query":"お好み焼きの作り方を教えて？詳しくね",
    "max_token":200,
     "temperature":1,
                }
start_time=time.time()
# FastAPIエンドポイントのURL
url = 'http://0.0.0.0:8000/generate/'  # FastAPIサーバーのURLに合わせて変更してください
# POSTリクエストを送信
response = requests.post(url, json=data)
print("genaraton time=",round(time.time()-start_time,2),"秒")
# レスポンスを表示
if response.status_code == 200:
    result = response.json()
    print("サーバーからの応答out:", result.get("out"))
else:
    print("リクエストが失敗しました。ステータスコード:", response.status_code)
