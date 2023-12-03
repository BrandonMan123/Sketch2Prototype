import pytest
from compute_clip import *
import clip
from torchmetrics.multimodal.clip_score import CLIPScore

def test_clip_cpu():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    image_1 = get_image("data/preferred/Page-9.png")
    text = "milk frother"
    assert compute_clip(image_1, text, metric) > 0

def test_clip_cuda():
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    image_1 = get_image("data/preferred/Page-9.png")
    text = "milk frother"
    assert compute_clip(image_1, text, metric) > 0

def test_image_clip_cpu():
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)
    assert compute_image_clip("data/preferred/Page-9.png", "data/preferred/Page-9.png", model, preprocess, device) > 0