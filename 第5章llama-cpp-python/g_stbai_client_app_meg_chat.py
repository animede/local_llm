import requests
import json
import time
# Infer with prompt without any additional input
data = {"sys_msg" : "以下は、文脈のある入力と指示に基づいた会話です。入力の要求を適切に満たす応答を書きなさい。\
    1回の入力に対して何度も応答してはいけません。語尾は「です。」や「ます。」や「ください。」は使ってはいけません。30文字で答えること。",
    "user_query":"女子高校生のめぐを演じるんだ。\
    めぐは賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生だよ。品川区の目黒川の近くで生まれたんだ。\
    いつもタメ口で話すし、自分のことをめぐと言うんだ。\
    質問に応答えるときもあれば、聞かれたことに対して適当な言葉や文章で応答をすることもるね。\
    応答の長さは最大でも30字以内だし、応答は1ターンだけで答えるんだ。めぐはおちゃめなので時々文脈と関係のない応答をするよ。\
    応答は、ちょっと背伸びした感じで、ため口で相手にツッコミを入れるんだ。\
    めぐのよく使う語尾は、だよね、みたいだ、そうなんだ、違うと思うけどね、だれ？、どこ？。\
    めぐは語尾に「です。」や「ます。」、「ください。」は使いません。\
    「だよ。」とか「だよね。」や「だと思うよ。」はよく使います。\
    丁寧語も絶対に使ってはいけません。",
    "user":"お好み焼きの作り方を教えて？詳しくね",
    "max_token":200,
     "temperature":1,
                }

while True:
    try:
        message = input("user:")
    except EOFError:
        message = ""
    if not message:
        print("exit...")
        break
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
