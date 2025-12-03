import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 参数设置
sar_dir = r"./data/Capella/SAR/"
rgb_dir = r"./data/Capella/RGB/"
out_sar_dir = r"./data/Capella_patches/sar/"
out_rgb_dir = r"./data/Capella_patches/rgb/"
tile_size = 1024
max_invalid_ratio = 0.3

os.makedirs(out_sar_dir, exist_ok=True)
os.makedirs(out_rgb_dir, exist_ok=True)

def get_patch_start_positions(length, tile_size):
    positions = list(range(0, length - tile_size + 1, tile_size))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions

def split_and_save(sar_path, rgb_path, base_name):
    try:
        with rasterio.open(sar_path) as sar_src, rasterio.open(rgb_path) as rgb_src:
            h, w = sar_src.height, sar_src.width
            count = 0

            y_positions = get_patch_start_positions(h, tile_size)
            x_positions = get_patch_start_positions(w, tile_size)

            for y in y_positions:
                for x in x_positions:
                    patch_name = f"{base_name}_{count:03d}.tif"
                    sar_out_path = os.path.join(out_sar_dir, patch_name)
                    rgb_out_path = os.path.join(out_rgb_dir, patch_name)

                    # ✅ 如果 patch 已存在，跳过
                    #if os.path.exists(sar_out_path) and os.path.exists(rgb_out_path):
                    #    count += 1
                    #    continue

                    window = Window(x, y, tile_size, tile_size)
                    sar_patch = sar_src.read(window=window)
                    rgb_patch = rgb_src.read(window=window).astype(np.uint8)

                    valid_ratio = np.count_nonzero(sar_patch) / sar_patch.size
                    if valid_ratio < (1 - max_invalid_ratio):
                        count += 1
                        continue

                    sar_profile = {
                        'driver': 'GTiff',
                        'height': tile_size,
                        'width': tile_size,
                        'count': sar_patch.shape[0],
                        'dtype': 'uint16'
                    }
                    with rasterio.open(sar_out_path, 'w', **sar_profile) as dst:
                        dst.write(sar_patch)

                    rgb_profile = {
                        'driver': 'GTiff',
                        'height': tile_size,
                        'width': tile_size,
                        'count': rgb_patch.shape[0],
                        'dtype': 'uint8'
                    }
                    with rasterio.open(rgb_out_path, 'w', **rgb_profile) as dst:
                        dst.write(rgb_patch)

                    count += 1

        return f"✅ 已处理: {base_name}"
    except Exception as e:
        return f"❌ 失败: {base_name} | 错误: {str(e)}"

def main():
    files = [f for f in os.listdir(sar_dir) if f.endswith(".tif") and os.path.exists(os.path.join(rgb_dir, f))]
    # print(len(files))
    tasks = []
    
    with ThreadPoolExecutor(max_workers=12) as executor:  # 你可以试试 max_workers=6
        for fname in files:
            sar_path = os.path.join(sar_dir, fname)
            rgb_path = os.path.join(rgb_dir, fname)
            base_name = os.path.splitext(fname)[0]
            future = executor.submit(split_and_save, sar_path, rgb_path, base_name)
            tasks.append(future)

        for f in tqdm(as_completed(tasks), total=len(tasks), desc="并行处理中", ncols=100):
            print(f.result())

if __name__ == "__main__":
    main()
