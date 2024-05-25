# 自分で作るAIキャラ　第1部ローカルLLM編
## 第2章
```
sudo apt install git
sudo apt install -y python3.11 python3.11-venv
git clone https://github.com/animede/local_llm.git
sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv llm
source ./llm/bin/activate
cd local_llm

pip install -r requirements.txt
```
## 第4章

### llama-cpp-pythonのビルド

githubの以下のリンクがコードです。

https://github.com/abetlen/llama-cpp-python

### #インストール

miniconda環境でインストール

引きつづき、llama.cppで作成した環境を使いたいのですが、うまくGPU版が構築できません。なのでconda系で仮想環境作成します。以下で最新版がインストールされます。Pythonは3.12.2

mkdir -p ~/miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

rm -rf ~/miniconda3/miniconda.sh

ライセンスに関する、たくさんのメッセージが出力されます。途中YesかNoか聞いてくるので、インストール先はY、conda initもYで良いと思います。

#### このあと環境構築

conda create -n llm

conda activate llm
