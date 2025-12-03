## 用于swinsimmim版本的下游任务测试
import torch
from pathlib import Path

# 输入路径：SimMIM 训练后的 checkpoint
orig_ckpt_path = "./work_dirs/mars_base/checkpoint-11.pth"
save_dir = Path("./mars_weights/mars_base")
save_dir.mkdir(parents=True, exist_ok=True)

# 加载 checkpoint
ckpt = torch.load(orig_ckpt_path, map_location='cpu',weights_only=False)['model']

# 要删除的多余键
remove_keys = ["norm.weight", "norm.bias", "head.fc.weight", "head.fc.bias"]
# remove_keys2 = []
def convert_state_dict(state_dict, prefix):
    out = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        new_k = k.replace(prefix, "")
        if new_k.startswith("layers."):
            parts = new_k.split(".")
            new_k = f"layers_{parts[1]}." + ".".join(parts[2:])
        if not any(new_k == rm for rm in remove_keys):
            out[new_k] = v
    return out

# 提取两个模型
rgb_weights = convert_state_dict(ckpt, "encoder_rgb.encoder.")
sar_weights = convert_state_dict(ckpt, "encoder_sar.encoder.")

# 保存为新的 .pth 文件
torch.save(rgb_weights, save_dir / "mars_rgb_encoder_only.pth")
torch.save(sar_weights, save_dir / "mars_sar_encoder_only.pth")
print("✅ 权重已保存：", save_dir)
