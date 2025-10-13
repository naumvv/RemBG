import sys
import base64
import mimetypes
from pathlib import Path
from html import escape

# какие исходные расширения считаем картинками
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def as_data_url(path: Path, force_mime: str | None = None) -> str:
    data = path.read_bytes()
    mime = force_mime or mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

def main(orig_dir, proc_dir, out_html):
    orig_dir = Path(orig_dir)
    proc_dir = Path(proc_dir)
    out_html = Path(out_html)

    # Собираем пары по имени файла (stem)
    originals = {
        p.stem: p for p in orig_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }
    processed = {
        p.stem: p for p in proc_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    }

    keys = sorted(set(originals) & set(processed))
    if not keys:
        print("❌ Нет совпадающих имён файлов (по stem).")
        sys.exit(1)

    rows_html = []
    for k in keys:
        o = originals[k]
        p = processed[k]
        try:
            o_url = as_data_url(o)                 # mime по расширению
            p_url = as_data_url(p, "image/png")    # явно PNG
        except Exception as e:
            print(f"⚠️  Пропускаю {k}: {e}")
            continue

        rows_html.append(f"""
        <div class="pair">
          <div class="card">
            <div class="label">Original</div>
            <img loading="lazy" src="{o_url}" alt="{escape(o.name)}">
            <div class="filename">{escape(o.name)}</div>
          </div>
          <div class="card checker">
            <div class="label">No background (PNG)</div>
            <img loading="lazy" src="{p_url}" alt="{escape(p.name)}">
            <div class="filename">{escape(p.name)}</div>
          </div>
        </div>
        """.strip())

    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Сравнение: оригинал vs без фона (inline)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --gap: 16px;
    --card-bg: #fff;
    --text: #111;
    --muted: #666;
    --border: #e5e5e5;
  }}
  body {{
    margin: 0 auto;
    padding: 24px;
    max-width: 1200px;
    font: 14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    color: var(--text);
    background: #fafafa;
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 22px;
  }}
  .meta {{
    color: var(--muted);
    margin-bottom: 20px;
  }}
  .pair {{
    display: grid;
    grid-template-columns: 1fr;
    gap: var(--gap);
    margin-bottom: var(--gap);
  }}
  @media (min-width: 900px) {{
    .pair {{
      grid-template-columns: 1fr 1fr;
    }}
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }}
  .checker {{
    background:
      conic-gradient(#ddd 25%, transparent 0 50%, #ddd 0 75%, transparent 0)
      0 0/20px 20px, var(--card-bg);
  }}
  .label {{
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .filename {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
    word-break: break-all;
  }}
  img {{
    display: block;
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    background: #fff;
  }}
  .note {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
  }}
</style>
</head>
<body>
  <h1>Сравнение: оригинал vs без фона</h1>
  <div class="meta">
    Встроенные изображения (data URL). Пар: {len(rows_html)}
  </div>

  {"".join(rows_html)}

  <p class="note">Шахматный фон справа помогает увидеть прозрачность PNG.</p>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"✅ Готово: {out_html.resolve()} (пар: {len(rows_html)})")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Использование: python generate_compare_inline_html.py <originals_dir> <no_bg_dir> <out.html>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])