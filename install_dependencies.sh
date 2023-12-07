echo "Installing CPU-based dependencies"
# ? source activate pytorch
pip install torch torchvision torchaudio
pip install transformers
pip install sentencepiece
pip install huggingface_hub
echo "Installed dependencies"
