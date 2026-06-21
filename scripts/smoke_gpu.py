from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ham_pipeline.classification_losses import ClassificationLossConfig, build_classification_loss
from ham_pipeline.losses import build_metric_loss
from ham_pipeline.methods import all_methods, get_method
from ham_pipeline.model import ConvNeXtMetric


assert torch.cuda.is_available(), "CUDA unavailable"
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
counts = torch.tensor([233, 354, 790, 91, 779, 4779, 100], device=device)

model = ConvNeXtMetric("convnext_base", pretrained=False).to(
    device, memory_format=torch.channels_last
).train()
images = torch.randn(4, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
targets = torch.tensor([0, 0, 1, 1], device=device)

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits, embeddings = model(images)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss = loss + build_metric_loss("supcon")(embeddings, targets)

    for mode in ("arcface", "cosface"):
        angular_logits, _ = model(images, labels=targets, classifier_mode=mode)
        angular_loss = torch.nn.functional.cross_entropy(angular_logits, targets)
        assert torch.isfinite(angular_loss), f"non-finite {mode}"

loss.backward()

synthetic_targets = torch.arange(14, device=device) % 7
synthetic_embeddings = torch.nn.functional.normalize(
    torch.randn(14, 256, device=device), dim=1
)
proxies = torch.randn(7, 256, device=device)

for method_name in all_methods():
    spec = get_method(method_name)
    classification = build_classification_loss(
        ClassificationLossConfig(name=spec.classification_loss), counts
    )
    value = classification(torch.randn(14, 7, device=device), synthetic_targets)
    assert torch.isfinite(value), f"non-finite classification loss for {method_name}"
    metric = build_metric_loss(spec.metric_loss)
    if metric is not None:
        value = metric(
            synthetic_embeddings,
            synthetic_targets,
            proxies=proxies,
            class_counts=counts,
        )
        assert torch.isfinite(value), f"non-finite metric loss for {method_name}"

print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "bf16": torch.cuda.is_bf16_supported(),
        "methods": len(all_methods()),
        "loss": float(loss.detach()),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
)
