from pathlib import Path
from PIL import Image
from rembg import new_session, remove
import argparse
import psutil
import os
import time

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def get_memory_usage_mb():
    """Возвращает текущее использование ОЗУ процессом в МБ."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)


def get_vram_usage_mb():
    """Возвращает использование видеопамяти (VRAM) в МБ."""
    if not GPU_AVAILABLE:
        return 0.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 * 1024)


def remove_background(src_img_path, output_dir):
    """Удаляет фон с изображения с помощью rembg, измеряет время и память."""
    print(f"\n📸 Processing: {src_img_path.name}")

    before_mem = get_memory_usage_mb()
    before_vram = get_vram_usage_mb()
    start_time = time.time()

    data = Image.open(src_img_path)
    model_name = "birefnet-general-lite"
    session = new_session(model_name)
    img = remove(data, session=session)

    out_path = output_dir / f"{src_img_path.stem}.png"
    img.save(out_path)

    end_time = time.time()
    after_mem = get_memory_usage_mb()
    after_vram = get_vram_usage_mb()

    print(f"✅ Saved: {out_path.name}")
    print(f"⏱ Time elapsed: {end_time - start_time:.2f} sec")
    print(f"💾 RAM delta: {after_mem - before_mem:.2f} MB (total {after_mem:.2f} MB)")
    if GPU_AVAILABLE:
        print(f"🎮 VRAM delta: {after_vram - before_vram:.2f} MB (total {after_vram:.2f} MB)")
    else:
        print("🎮 VRAM info: unavailable (no NVIDIA GPU or pynvml not installed)")
    print("-" * 50)

    return out_path


def process_folder(input_dir, output_dir):
    """Обрабатывает все изображения из input_dir и сохраняет PNG в output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        try:
            remove_background(img_path, output_dir)
        except Exception as e:
            print(f"❌ Error with {img_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch background remover using rembg with memory/time tracking"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output folder to save PNGs")
    args = parser.parse_args()

    process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
