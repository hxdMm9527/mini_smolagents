"""导出 bge-small-zh-v1.5 到 ONNX（一次性，需 torch + transformers）。

用法：python scripts/export_bge_onnx.py
产物：models/bge-small-zh-v1.5.onnx + models/tokenizer.json
说明：运行期 embedding 优先走 ONNX（启动快 ~1.7s，无需 torch）；
     若无产物则自动回退 sentence_transformers（启动慢 ~8s）。
"""
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-small-zh-v1.5"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"


class BGEWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        out = self.m(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, p=2, dim=1)


def main():
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

    OUT.mkdir(exist_ok=True)
    torch.onnx.export(
        BGEWrapper(model),
        (torch.tensor([[0, 1, 2]]), torch.tensor([[1, 1, 1]])),
        str(OUT / "bge-small-zh-v1.5.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
        opset_version=14,
        dynamo=False,
    )
    src = tok.vocab_file
    if not src:
        hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots"
        cands = sorted(hub.glob("*/tokenizer.json")) if hub.is_dir() else []
        src = str(cands[0]) if cands else None
    if src:
        shutil.copy(src, OUT / "tokenizer.json")
    print("exported:", OUT / "bge-small-zh-v1.5.onnx")


if __name__ == "__main__":
    main()
