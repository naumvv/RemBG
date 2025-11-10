from pathlib import Path
from PIL import Image
from rembg import new_session, remove
import argparse
import psutil
import os
import time
import math
from rembg.sessions import sessions_class
import onnxruntime as ort
from typing import Optional, Type
from rembg.sessions.base import BaseSession


try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def get_memory_usage_mb():
    """Текущее использование ОЗУ процессом в МБ."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)


def get_vram_usage_mb():
    """Текущее использование VRAM (в МБ) для GPU:0."""
    if not GPU_AVAILABLE:
        return 0.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 * 1024)


def percentile(data, p):
    """p-перцентиль (0..100) без внешних библиотек (линейная интерполяция)."""
    if not data:
        return 0.0
    xs = sorted(data)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def remove_background(src_img_path, output_dir):
    """Удаляет фон, измеряет время и дельты RAM/VRAM (без пиков)."""
    print(f"\n📸 Processing: {src_img_path.name}")

    before_mem = get_memory_usage_mb()
    before_vram = get_vram_usage_mb()
    start_time = time.time()

    data = Image.open(src_img_path)

    session_class: Optional[Type[BaseSession]] = None
    model_name = "birefnet-general-lite"

    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    rembg_session = session_class(model_name, sess_opts)

    # Удаляем фон
    img = await remove(image, session=rembg_session)

    out_path = output_dir / f"{src_img_path.stem}.png"
    img.save(out_path)

    end_time = time.time()
    after_mem = get_memory_usage_mb()
    after_vram = get_vram_usage_mb()

    elapsed = end_time - start_time
    ram_delta = after_mem - before_mem
    vram_delta = after_vram - before_vram

    print(f"✅ Saved: {out_path.name}")
    print(f"⏱ Time elapsed: {elapsed:.2f} sec")
    print(f"💾 RAM delta: {ram_delta:.2f} MB (total {after_mem:.2f} MB)")
    if GPU_AVAILABLE:
        print(f"🎮 VRAM delta: {vram_delta:.2f} MB (total {after_vram:.2f} MB)")
    else:
        print("🎮 VRAM info: unavailable (no NVIDIA GPU or pynvml not installed)")
    print("-" * 50)

    return elapsed, ram_delta, vram_delta


def process_folder(input_dir, output_dir):
    """Обрабатывает все изображения и печатает p95, p99 и max по метрикам."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = []
    ram_deltas = []
    vram_deltas = []
    count = 0

    for img_path in input_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        try:
            elapsed, ram_delta, vram_delta = remove_background(img_path, output_dir)
            times.append(elapsed)
            ram_deltas.append(ram_delta)
            vram_deltas.append(vram_delta)
            count += 1
        except Exception as e:
            print(f"❌ Error with {img_path.name}: {e}")

    if count > 0:
        # p95 / p99 / max для времени
        p95_time = percentile(times, 95)
        p99_time = percentile(times, 99)
        max_time = max(times)

        # p95 / p99 / max для RAM Δ
        p95_ram = percentile(ram_deltas, 95)
        p99_ram = percentile(ram_deltas, 99)
        max_ram = max(ram_deltas)

        print("\n📊 ==== SUMMARY ====")
        print(f"🖼 Processed images: {count}")
        print(f"⏱ Time per image — p95: {p95_time:.2f}s | p99: {p99_time:.2f}s | max: {max_time:.2f}s")
        print(f"💾 RAM Δ (MB)     — p95: {p95_ram:.2f} | p99: {p99_ram:.2f} | max: {max_ram:.2f}")

        if GPU_AVAILABLE:
            p95_vram = percentile(vram_deltas, 95)
            p99_vram = percentile(vram_deltas, 99)
            max_vram = max(vram_deltas)
            print(f"🎮 VRAM Δ (MB)    — p95: {p95_vram:.2f} | p99: {p99_vram:.2f} | max: {max_vram:.2f}")
        else:
            print("🎮 VRAM stats: unavailable (no NVIDIA GPU or pynvml not installed)")
        print("====================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch background remover using rembg with p95/p99/max stats"
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output folder to save PNGs")
    args = parser.parse_args()

    process_folder(args.input, args.output)

    # Опционально: корректное завершение NVML
    if GPU_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
