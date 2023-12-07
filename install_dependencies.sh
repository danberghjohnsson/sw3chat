echo "Installing CPU-based dependencies"
# ? source activate pytorch
pip install torch torchvision torchaudio
pip install transformers
pip install sentencepiece
echo "Installed dependencies"
