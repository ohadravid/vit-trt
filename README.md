
# Download Weights

Download the distilled checkpoint by running:

```bash
wget https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_s_k710_dl_from_giant.pth
```

# Commands

```bash
# Run inference
uv run python main.py infer

# Export the model as an onnx file
uv run python main.py export_onnx
# and run inference using it
uv run python main.py infer_onnx
```