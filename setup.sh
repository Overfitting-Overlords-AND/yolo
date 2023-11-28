mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

export PATH=~/miniconda3/bin:~/miniconda3/scripts:$PATH

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

exit

bash conda install python=3.11.5

conda create -n yolo
conda activate yolo
conda install cudatoolkit=11.0 -y

pip install torch sentencepiece datasets bitsandbytes peft accelerate scipy pipx wandb uvicorn
pip install huggingface_hub=0.2.1
pip install huggingface_hub[torch]

exit
