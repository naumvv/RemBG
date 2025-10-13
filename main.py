from pathlib import Path
from PIL import Image
from rembg import new_session, remove
import argparse
import psutil  # <--- добавлено
import os

def get_memory_usage_mb():
    """Возвращает текущее использование памяти процессом в МБ."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    return mem_bytes / (1024 * 1024)


def remove_background(src_img_path, output_dir):
    """Удаляет фон с изображения с помощью rembg и замеряет память."""
    print(f"\n📸 Processing: {src_img_path.name}")
    before_mem = get_memory_usage_mb()

    data = Image.open(src_img_path)
    model_name = "birefnet-general-lite"
    session = new_session(model_name)
    img = remove(data, session=session)

    out_path = output_dir / f"{src_img_path.stem}.png"
    img.save(out_path)

    after_mem = get_memory_usage_mb()
    print(f"✅ Saved: {out_path.name}")
    print(f"💾 Memory used: {after_mem - before_mem:.2f} MB (Δ)")
    print(f"📊 Total memory usage: {after_mem:.2f} MB\n")

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
        description="Batch background remover using rembg"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input folder with images"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output folder to save PNGs"
    )
    args = parser.parse_args()

    process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
