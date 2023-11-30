bash conda install python=3.11.5

conda create -n yolo
conda activate yolo
conda install cudatoolkit=11.0 -y

pip install torch sentencepiece datasets bitsandbytes peft accelerate scipy pipx wandb uvicorn
pip install huggingface_hub=0.2.1
pip install huggingface_hub[torch]

exit
