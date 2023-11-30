bash conda install python=3.11.5

source conda_activate.sh
conda install cudatoolkit=11.0 -y

pip install torch pipx wandb uvicorn torchvision matplotlb

exit
