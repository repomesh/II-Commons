# Installs dependencies including the dev version of transformers and accelerate from requirements.txt
pip install -r requirements.txt
# mac only:
# https://github.com/pytorch/pytorch/issues/77764
# pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
