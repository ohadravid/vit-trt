import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import io
import sys
from pathlib import Path

import onnxruntime as ort
from torchvision.io import read_video
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from video_mae import vit_small_patch16_224, VisionTransformer


class ResizeVideoToLength(nn.Module):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def forward(self, x):
        T, C, H, W = x.shape
        frame_idxs = torch.linspace(0, T - 1, steps=self.length, device=x.device)
        frame_idxs = torch.round(frame_idxs).clamp(0, T - 1).long()
        x = x[frame_idxs]

        return x


MAX_SEQ_LEN = 96


def get_val_transform(
    image_wh: tuple[int, int],
    norm_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    norm_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> Tuple[T.Compose, T.Compose]:
    return T.Compose(
        [
            T.Resize(image_wh, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=norm_mean, std=norm_std),
            ResizeVideoToLength(MAX_SEQ_LEN),
        ]
    )


class VideoMAEBinSearch(nn.Module):
    def __init__(self, model: VisionTransformer):
        super().__init__()
        self.model = model

    def forward(self, x):
        B, T, C, H, W = x.shape

        # The model expects [B, C, T, H, W], with T=model.all_frames (16).
        x = x.reshape(-1, self.model.all_frames, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        cls = self.model(x)

        cls = cls.reshape(B, -1, cls.shape[-1])
        cls = cls.mean(dim=1)

        cls = F.softmax(cls, dim=1)
        
        return cls


def get_labels_and_video():
    transform = get_val_transform(image_wh=(224, 224))

    video = read_video("tea.mp4", pts_unit="sec", output_format="TCHW")[0]
    video = transform(video)

    labels = Path("./label_map_k710.txt").read_text().splitlines()
    return labels, video


def get_model():
    backbone_ckpt = "./vit_s_k710_dl_from_giant.pth"

    video_mae_model = vit_small_patch16_224(num_classes=710)

    print(f"Loading pretrained backbone from {backbone_ckpt}")
    ckpt = torch.load(backbone_ckpt, map_location="cpu", weights_only=True)["module"]

    video_mae_model.load_state_dict(ckpt, strict=False)

    model = VideoMAEBinSearch(video_mae_model)

    model = model.eval()

    return model


@torch.inference_mode()
def infer():
    model = get_model()
    labels, video = get_labels_and_video()
    video_as_batch = video.unsqueeze(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    video_as_batch = video_as_batch.to(device)

    import time
    start_time = time.perf_counter()
    n_runs = 3
    for _ in range(n_runs):
        cls = model(video_as_batch)
    end_time = time.perf_counter()
    print(f"Inference runs per sec: {n_runs / (end_time - start_time):.2f} on {device}")

    top_cls = torch.topk(cls.squeeze(0), 3)

    for cls_idx, score in zip(top_cls.indices, top_cls.values):
        print(f"{labels[cls_idx]}: {score:.2f}")


def export_onnx():
    model = get_model()
    labels, video = get_labels_and_video()
    video_as_batch = video.unsqueeze(0)

    onnx_bytes = io.BytesIO()

    torch.onnx.export(
        model,
        (video_as_batch,),
        onnx_bytes,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["video"],
        output_names=["cls"],
    )

    Path("model.onnx").write_bytes(onnx_bytes.getvalue())


def infer_onnx():
    labels, video = get_labels_and_video()
    video_as_batch = video.unsqueeze(0)

    model_onnx = Path("model.onnx")
    ort_sess = ort.InferenceSession(model_onnx.read_bytes())

    video_as_batch = video_as_batch.numpy()
    cls, *_ = ort_sess.run(
        None,
        {
            "video": video_as_batch,
        },
    )

    cls = torch.from_numpy(cls)
    top_cls = torch.topk(cls.squeeze(0), 3)

    for cls_idx, score in zip(top_cls.indices, top_cls.values):
        print(f"{labels[cls_idx]}: {score:.2f}")


def infer_trt():
    import torch_tensorrt

    labels, video = get_labels_and_video()
    video_as_batch = video.unsqueeze(0).cuda()

    model = torch_tensorrt.runtime.PythonTorchTensorRTModule(
        Path("model.trt").read_bytes(),
        input_binding_names=[
            "video",
        ],
        output_binding_names=[
            "cls",
        ],
    )

    import time
    start_time = time.perf_counter()
    n_runs = 30
    for _ in range(n_runs):
        cls = model(video_as_batch)
    end_time = time.perf_counter()
    print(f"Inference runs per sec: {n_runs / (end_time - start_time):.2f}")

    top_cls = torch.topk(cls.squeeze(0), 3)
    
    for cls_idx, score in zip(top_cls.indices, top_cls.values):
        print(f"{labels[cls_idx]}: {score:.2f}")


def main():
    action = sys.argv[1]

    if action == "infer":
        infer()
    elif action == "export_onnx":
        export_onnx()
    elif action == "infer_onnx":
        infer_onnx()
    elif action == "infer_trt":
        infer_trt()
    else:
        raise ValueError(f"Unknown action {action}")


if __name__ == "__main__":
    main()
