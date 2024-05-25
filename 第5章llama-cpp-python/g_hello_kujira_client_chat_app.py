from g_hello_kujira_gguf_client import  TextGenerator,Communicator
import time
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

