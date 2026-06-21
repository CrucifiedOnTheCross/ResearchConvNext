from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from ham_pipeline.losses import build_metric_loss
from ham_pipeline.model import ConvNeXtMetric

assert torch.cuda.is_available(), "CUDA unavailable"
torch.set_float32_matmul_precision("high")
device=torch.device("cuda")
model=ConvNeXtMetric("convnext_base",pretrained=False).to(device,memory_format=torch.channels_last).train()
x=torch.randn(4,3,224,224,device=device).to(memory_format=torch.channels_last)
y=torch.tensor([0,0,1,1],device=device)
with torch.amp.autocast("cuda",dtype=torch.bfloat16):
    logits,z=model(x)
    loss=torch.nn.functional.cross_entropy(logits,y)+build_metric_loss("supcon")(z,y,proxies=model.proxies)
loss.backward()
for name in ("supcon","triplet","n_pairs","multi_similarity","circle","proxy_anchor","arcface","cosface","center","paco","bcl","sbcl","prototype","meta_prototype"):
    value=build_metric_loss(name)(z.detach(),y,proxies=model.proxies.detach())
    assert torch.isfinite(value), f"non-finite {name}"
print({"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"bf16":torch.cuda.is_bf16_supported(),"loss":float(loss),"peak_vram_mib":round(torch.cuda.max_memory_allocated()/2**20)})
