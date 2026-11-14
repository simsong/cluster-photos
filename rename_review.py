"""Build a review page for proposed renames (e.g. slide captions from a catalog) and emit approved names as JSON.

Theory of purpose: renaming photos to carry their descriptions is hard to undo by hand, so each proposed name is
shown beside its photo for a person to accept, edit or skip before anything changes.
Theory of operation: reads a plan CSV (path, new_name, note, unsure). It writes 400 px thumbnails, then a static
page where each card has an editable name field (prefilled), a skip box and the note; unsure rows are highlighted.
Edits persist in localStorage, and Copy JSON emits {"plan": <hash of the plan CSV>, "names": {id: name},
"skip": [ids]} with only rows that differ from the proposal. apply_renames.py checks the hash and renames.
Read-only on photos.
"""

import argparse
import csv
import hashlib
import html
import os
import sys
from multiprocessing import Pool

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


def rid(path):
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def thumb(args):
    path, tdir = args
    name = rid(path) + ".jpg"
    dest = os.path.join(tdir, name)
    if not os.path.exists(dest):
        register_heif_opener()
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img.thumbnail((400, 400))
        img.save(dest, quality=82)
    return path, "thumbs/" + name


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan", help="CSV with columns path,new_name,note,unsure")
    ap.add_argument("--out", required=True, help="directory for the page and thumbnails")
    ap.add_argument("--title", default="Rename review")
    args = ap.parse_args()
    with open(args.plan, "rb") as f:
        plan_id = hashlib.sha256(f.read()).hexdigest()[:12]
    with open(args.plan, newline="") as f:
        rows = list(csv.DictReader(f))
    tdir = os.path.join(args.out, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    with Pool(8) as pool:
        thumbs = dict(pool.map(thumb, [(r["path"], tdir) for r in rows]))
    cards = "".join(
        f'<div class="card{" unsure" if r["unsure"] == "1" else ""}" data-id="{rid(r["path"])}">'
        f'<img loading="lazy" src="{thumbs[r["path"]]}">'
        f'<div class="old">{html.escape(os.path.basename(r["path"]))}</div>'
        f'<input type="text" data-id="{rid(r["path"])}" data-def="{html.escape(r["new_name"])}" value="{html.escape(r["new_name"])}">'
        f'<label><input type="checkbox" data-skip="{rid(r["path"])}"> skip (do not rename)</label>'
        f'<div class="note">{"UNSURE: " if r["unsure"] == "1" else ""}{html.escape(r["note"])}</div></div>' for r in rows)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#fafafa;--fg:#222;--card:#fff;--muted:#666;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151515;--fg:#eee;--card:#222;--muted:#aaa;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;margin:0 16px 60px}}
.bar{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:2;display:flex;gap:12px;align-items:center}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}}
.card{{background:var(--card);border:1px solid var(--line);border-left:6px solid #2e9e4f;padding:8px}} .card.unsure{{border-left-color:#d9a400}}
.card.skip{{opacity:.45}} .card.edited{{border-left-color:#2f6fd6}}
.card img{{width:100%;height:240px;object-fit:contain;background:#808080}} .old{{font-size:11px;color:var(--muted)}}
input[type=text]{{width:100%;font:inherit;font-weight:600;margin:4px 0;background:var(--bg);color:var(--fg);border:1px solid var(--line)}}
.note{{font-size:11px;color:var(--muted)}} .unsure .note{{color:#a15c00;font-weight:600}} textarea{{width:100%}}
</style></head><body>
<div class="bar"><b>{html.escape(args.title)}</b><span>{len(rows)} photos</span><span id="count"></span><button id="export">Copy JSON</button><span id="status"></span></div>
<div id="out" hidden><textarea id="json" readonly rows="4"></textarea></div>
<p>Each photo shows its proposed new name, which you can edit. Yellow means my match is unsure. Tick <b>skip</b> to leave a file as it is.
When done, press Copy JSON and save the text to a file, such as review.json. Unedited rows are renamed as proposed.</p>
<div class="grid">{cards}</div>
<script>
const PLAN = "{plan_id}", KEY = "rename-review-" + PLAN;
let D = {{plan: PLAN, names: {{}}, skip: []}};
try {{ D = Object.assign(D, JSON.parse(localStorage.getItem(KEY) || "{{}}")); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(D)); }} catch (e) {{}} }};
function paint() {{
  document.querySelectorAll(".card").forEach(c => {{
    const id = c.dataset.id, t = c.querySelector("input[type=text]");
    if (D.names[id] !== undefined) t.value = D.names[id];
    c.querySelector("input[type=checkbox]").checked = D.skip.includes(id);
    c.classList.toggle("skip", D.skip.includes(id)); c.classList.toggle("edited", D.names[id] !== undefined);
  }});
  document.getElementById("count").textContent = Object.keys(D.names).length + " edited · " + D.skip.length + " skipped";
}}
document.addEventListener("input", e => {{
  const el = e.target;
  if (el.type === "text") {{ if (el.value === el.dataset.def) delete D.names[el.dataset.id]; else D.names[el.dataset.id] = el.value; }}
  if (el.type === "checkbox") {{ const id = el.dataset.skip; D.skip = D.skip.filter(x => x !== id); if (el.checked) D.skip.push(id); }}
  paint(); save();
}});
paint();
document.getElementById("export").onclick = () => {{
  const t = document.getElementById("json"), j = JSON.stringify(D);
  document.getElementById("out").hidden = false; t.value = j; t.select();
  (navigator.clipboard ? navigator.clipboard.writeText(j) : Promise.reject()).then(
    () => document.getElementById("status").textContent = "copied: save it to a file such as review.json",
    () => document.getElementById("status").textContent = "select the text above and copy it");
}};
</script></body></html>"""
    with open(os.path.join(args.out, "rename_review.html"), "w") as f:
        f.write(page)
    print(f"plan {plan_id}: {len(rows)} rows -> {os.path.join(args.out, 'rename_review.html')}")


if __name__ == "__main__":
    sys.exit(main())
