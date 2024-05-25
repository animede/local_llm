from g_hello_stb_gguf_class import TextGenerator
import time

def main():
    generator = TextGenerator(512)

    while True:
        try:
            message = input("user: ")
        except EOFError:
            message = ""
        if not message:
            print("exit...")
            break

        response, gen_time = generator.generate_response(message)
        if response == "exit...":
            break

        print("generation time=", gen_time, "秒")
        print("final ans", response)

if __name__ == "__main__":
    main()

