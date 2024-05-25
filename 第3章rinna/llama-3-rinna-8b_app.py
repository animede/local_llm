from llama_3_rinna_8b_class import TextGenerator

# メイン関数
def main():
    #TextGeneratorクラスの初期化、モデルのロード
    text_generator = TextGenerator()
    system="""あなたは女子高校生の「めぐ」です。賢くて、おちゃめで、少しボーイッシュ、天真爛漫で好奇心旺盛な女子高生です。
            品川区の目黒川の近くで生まれました。いつも話言葉を使います。第一人称は「めぐ」と言いってください。「めぐ」のよく使う口癖は次のとおりです。
            その口癖に合わせた感じで話してください。みたいだ。そうなんだ。違うと思うけどね。だれ？。どこ？。userの質問にこたえなさい"""
    max_log=3
    log_list=[]
    while True:
        try:
            message = input("user:")
        except EOFError:
            message = ""
        if not message:
            break
        log_txt=result = ','.join(log_list)
        prpmpt = system + log_txt +message
        generated_text = text_generator.generate_text(message)
        #会話を記録
        log_list.append("user:"+message)
        log_list.append("めぐ:"+generated_text)
        if len(log_list)>max_log*2:
            del log_list[0] #userのメッセージ
            del log_list[0] #めぐのメッセージ
        print(log_list)
        print("めぐ:", generated_text)

if __name__ == "__main__":
    main()
