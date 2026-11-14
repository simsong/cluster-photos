"""Cluster the remaining photos by CLIP embedding and build a cluster-level review page (largest first).

Theory of purpose: per-image classification (receipt/screenshot) proved unreliable; Simson prefers to judge
groups of similar images at once (keep all, delete all, move to ~/Documents, or decide per image).
Theory of operation: a k-nearest-neighbour graph (K neighbours, cosine >= START) is built on MPS from
embeddings.sqlite3 (existing files only, never inside .photoslibrary). Its connected components are split
recursively at rising thresholds (STEPS) while a component is larger than MAX_SIZE or loose (some member has
cosine < COHESION to its centroid), which prevents chaining. Clusters of size >= 2 are ranked by size. The
page shows the first --top clusters, with a lightbox for full-size viewing (use serve_review.py so every browser can show the originals) that never leaves the page, so marks
are never lost. Within each shown cluster, resolution_dupes.py finds lower-resolution copies. Marks persist in localStorage, and "Copy JSON" emits only cluster choices and per-image
exceptions, stamped with a run id (a hash of clusters.csv). apply_review.py uses clusters.csv to act on the marks. Read-only on photos.
"""

import argparse
import collections
import csv
import hashlib
import html
import os
import sqlite3
from multiprocessing import Pool

import numpy as np
import torch

from PIL import Image, ImageFile, ImageOps
from pillow_heif import register_heif_opener

from resolution_dupes import find_lower_res

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
K, START, STEPS, MAX_SIZE, COHESION = 20, 0.90, (0.92, 0.94, 0.96, 0.98), 300, 0.82


def knn_graph(M, k=K, start=START, block=4096):
    """Symmetric neighbor lists from exact K-nearest-neighbor search over unit row vectors M."""
    n = M.shape[0]
    nbr = [[] for _ in range(n)]
    for s in range(0, n, block):
        v, i = torch.topk(M[s:s + block] @ M.T, min(k + 1, n), dim=1)  # + 1 because each image finds itself
        v, i = v.cpu().numpy(), i.cpu().numpy()
        for r in range(v.shape[0]):
            a = s + r
            for b, x in zip(i[r], v[r]):
                if b != a and x >= start:
                    nbr[a].append((int(b), float(x)))
                    nbr[int(b)].append((a, float(x)))
    return nbr


def components(nodes, nbr, thr):
    """Connected components of the subgraph on nodes with edges >= thr."""
    nodes = set(nodes)
    seen, out = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y, s in nbr[x]:
                if s >= thr and y in nodes and y not in seen:
                    seen.add(y)
                    stack.append(y)
        out.append(comp)
    return out


def split(comp, nbr, E, level):
    c = E[comp].mean(0)
    c /= np.linalg.norm(c)
    loose = (E[comp] @ c).min() < COHESION
    if len(comp) <= 1 or (len(comp) <= MAX_SIZE and not loose) or level >= len(STEPS):
        return [comp]
    out = []
    for sub in components(comp, nbr, STEPS[level]):
        out += split(sub, nbr, E, level + 1)
    return out


def iid(path):
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def thumb(path, tdir):
    """Write a 200 px JPEG thumbnail, keyed on path and size so changed files get a new one."""
    name = hashlib.sha1(f"{path}|{os.stat(path).st_size}".encode()).hexdigest()[:16] + ".jpg"
    dest = os.path.join(tdir, name)
    if not os.path.exists(dest):
        try:
            register_heif_opener()
            img = Image.open(path)
            img.draft("RGB", (400, 400))
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((200, 200))
            img.save(dest, quality=75)
        except Exception:
            return ""
    return "thumbs/" + name


def _thumb(args):
    return args[0], thumb(*args)


def build(root, db_path, out, top_n=500):
    """Cluster the embedded images under root and write clusters.csv and cluster_review.html to out."""
    PHOTOS = os.path.join(os.path.abspath(os.path.expanduser(root)), "")
    tdir = os.path.join(out, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    db = sqlite3.connect(db_path)
    rows = [(p, e, w, h) for p, e, w, h in db.execute("SELECT path, emb, width, height FROM images WHERE emb IS NOT NULL")
            if p.startswith(PHOTOS) and os.path.exists(p) and ".photoslibrary/" not in p]
    paths = [r[0] for r in rows]
    dims = {r[0]: f"{r[2]}×{r[3]}" for r in rows}
    E = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
    n = len(paths)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    M = torch.from_numpy(E).to(dev)
    nbr = knn_graph(M)
    clusters = []
    for comp in components(range(n), nbr, START):
        clusters += split(comp, nbr, E, 0)
    clusters = sorted((c for c in clusters if len(c) >= 2), key=len, reverse=True)
    print(f"{n:,} images; {len(clusters):,} clusters of 2+ covering {sum(map(len, clusters)):,} images; "
          f"largest {[len(c) for c in clusters[:10]]}", flush=True)

    pixels = [(r[2] or 0) * (r[3] or 0) for r in rows]
    lower = find_lower_res(clusters[:top_n], paths, E, pixels, os.path.join(out, "same_image_cache.json"))
    print(f"{len(lower):,} lower-resolution duplicates in the top {top_n} clusters", flush=True)
    with open(os.path.join(out, "clusters.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "id", "path", "lower_res_of"])
        for k, c in enumerate(clusters, 1):
            for x in c:
                w.writerow([k, iid(paths[x]), paths[x], iid(paths[lower[x]]) if x in lower else ""])

    with open(os.path.join(out, "clusters.csv"), "rb") as f:
        run_id = hashlib.sha256(f.read()).hexdigest()[:12]  # ties review marks to this exact clustering
    top = clusters[:top_n]
    with Pool(12) as pool:
        thumbs = dict(pool.map(_thumb, [(paths[x], tdir) for c in top for x in c], chunksize=16))

    def cluster_html(k, c):
        c = sorted(c, key=lambda x: paths[x])
        cen = E[c].mean(0)
        cen /= np.linalg.norm(cen)
        coh = float((E[c] @ cen).mean())
        folders = collections.Counter(os.path.dirname(paths[x]).removeprefix(PHOTOS) for x in c).most_common(3)
        fold = "; ".join(f"{html.escape(f)} ({m})" for f, m in folders)
        n_low = sum(x in lower for x in c)
        choices = [("keep", "Keep all"), ("delete", "Delete all"), ("move", "Move all to ~/Documents"), ("each", "Decide per image")]
        if n_low:
            choices.insert(1, ("dedupe", f"Keep highest resolution (delete {n_low} lower-res)"))
        acts = "".join(f'<label><input type="radio" name="c{k}" value="{v}" data-c="{k}"{" checked" if v == "keep" else ""}> {t}</label>'
                       for v, t in choices)
        keepers = {lower[x] for x in c if x in lower}
        tag = lambda x: (f'<div class="res low">lower-res copy ({dims[paths[x]]})</div>' if x in lower
                         else f'<div class="res high">highest res ({dims[paths[x]]})</div>' if x in keepers else "")
        imgs = "".join(
            f'<figure class="im" data-id="{iid(paths[x])}" data-c="{k}"{" data-lower=1" if x in lower else ""}><img loading="lazy" src="{thumbs.get(paths[x], "")}" '
            f'data-full="file://{html.escape(paths[x])}" data-path="{html.escape(paths[x].removeprefix(PHOTOS))}" data-dims="{dims[paths[x]]}">'
            f'{tag(x)}<figcaption>{html.escape(os.path.basename(paths[x]))}</figcaption>'
            f'<select data-id="{iid(paths[x])}" data-c="{k}"><option value="">(cluster choice)</option><option value="keep">keep</option>'
            f'<option value="delete">delete</option><option value="move">move to Documents</option></select></figure>' for x in c)
        return (f'<section class="cl" id="cl{k}" data-low="{n_low}"><h2>#{k} · {len(c)} images{f" · {n_low} lower-res copies" if n_low else ""} · cohesion {coh:.3f}</h2><div class="m">{fold}</div>'
                f'<div class="acts">{acts}</div><div class="grid">{imgs}</div></section>')

    body = "".join(cluster_html(k, c) for k, c in enumerate(top, 1))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Photo Clusters</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#fafafa;--fg:#222;--card:#fff;--muted:#666;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151515;--fg:#eee;--card:#222;--muted:#aaa;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;margin:0 16px 60px}}
.bar{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:3;display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
.cl{{background:var(--card);border:1px solid var(--line);margin:10px 0;padding:8px;border-left:6px solid #2e9e4f}}
.cl.delete{{border-left-color:#d23c3c}} .cl.move{{border-left-color:#2f6fd6}} .cl.each{{border-left-color:#a15c00}} .cl.dedupe{{border-left-color:#8a4fd0}}
.res{{font-size:10px;font-weight:700}} .res.low{{color:#d23c3c}} .res.high{{color:#2e9e4f}}
.cl h2{{margin:2px 0;font-size:15px}} .m{{color:var(--muted);font-size:12px}} .acts label{{margin-right:12px;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px;margin-top:6px}}
.im{{margin:0;padding:3px;border-radius:4px;outline:3px solid transparent}} .im img{{width:100%;height:120px;object-fit:contain;cursor:zoom-in}}
.im.delete{{outline-color:#d23c3c}} .im.delete img{{opacity:.55}} .im.move{{outline-color:#2f6fd6}} .im.keep{{outline-color:#2e9e4f}}
figcaption{{font-size:10px;word-break:break-all;color:var(--muted)}} select{{width:100%;font-size:11px}}
#lb{{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;z-index:9;align-items:center;justify-content:center;flex-direction:column;color:#eee}}
#lb img{{max-width:96vw;max-height:88vh;object-fit:contain}} #lb.on{{display:flex}} #lbcap{{font-size:12px;margin-top:6px}}
textarea{{width:100%}}
</style></head><body>
<div class="bar"><b>Photo clusters</b> <span>{len(top)} largest of {len(clusters):,} clusters ({n:,} images)</span>
<span id="count"></span><button id="alldedupe" title="Applies to clusters still set to Keep all">Keep highest resolution in all clusters ({len(lower):,} lower-res)</button><button id="export">Copy JSON</button><span id="status"></span></div>
<div id="out" hidden><textarea id="json" readonly rows="4"></textarea></div>
<p><b>Keep highest resolution</b> deletes only files confirmed to be the same uncropped picture as a larger file in the cluster.
Every cluster defaults to <b>Keep all</b>, so nothing happens unless you choose. Pick one action per cluster. With <b>Decide per image</b>,
or to make an exception, use the menu under any photo. Click a photo to see it full size here (Esc or click to close), and arrow keys step
through its cluster. Your marks are saved in this browser as you go. When done, press Copy JSON and save the text to a file, such as review.json.</p>
{body}
<div id="lb"><img id="lbimg"><div id="lbcap"></div></div>
<script>
const RUN = "{run_id}", KEY = "cluster-review-" + RUN;
let D = {{run: RUN, c: {{}}, x: {{}}}};
try {{ D = Object.assign(D, JSON.parse(localStorage.getItem(KEY) || "{{}}")); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(D)); }} catch (e) {{}} }};
function paint(k) {{
  const sec = document.getElementById("cl" + k), a = D.c[k] || "keep";
  sec.classList.remove("keep", "delete", "move", "each", "dedupe"); sec.classList.add(a);
  sec.querySelectorAll(".im").forEach(f => {{
    const x = D.x[f.dataset.id] || (a === "each" ? "" : a === "dedupe" ? (f.dataset.lower ? "delete" : "keep") : a);
    f.classList.remove("keep", "delete", "move"); if (x) f.classList.add(x);
    f.querySelector("select").value = D.x[f.dataset.id] || "";
  }});
  const r = sec.querySelector(`input[value="${{a}}"]`); if (r) r.checked = true;
}}
function count() {{
  document.getElementById("count").textContent = Object.keys(D.c).length + " clusters set · " + Object.keys(D.x).length + " photo exceptions";
}}
document.addEventListener("change", e => {{
  const el = e.target;
  if (el.type === "radio" && el.dataset.c) {{ if (el.value === "keep") delete D.c[el.dataset.c]; else D.c[el.dataset.c] = el.value; paint(el.dataset.c); }}
  else if (el.tagName === "SELECT") {{ if (el.value) D.x[el.dataset.id] = el.value; else delete D.x[el.dataset.id]; paint(el.dataset.c); }}
  count(); save();
}});
let cur = null;
const lb = document.getElementById("lb"), lbimg = document.getElementById("lbimg"), cap = document.getElementById("lbcap");
function show(img) {{ cur = img; lbimg.src = location.protocol.startsWith("http") ? "photo/" + img.closest(".im").dataset.id : img.dataset.full; cap.textContent = img.dataset.path + " · " + img.dataset.dims; lb.classList.add("on"); }}
document.addEventListener("click", e => {{
  if (e.target.matches(".im img")) {{ e.preventDefault(); show(e.target); }}
  else if (lb.classList.contains("on") && e.target.closest("#lb")) lb.classList.remove("on");
}});
document.addEventListener("keydown", e => {{
  if (!lb.classList.contains("on")) return;
  if (e.key === "Escape") lb.classList.remove("on");
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {{
    const all = [...cur.closest(".grid").querySelectorAll("img")], i = all.indexOf(cur);
    const j = (i + (e.key === "ArrowRight" ? 1 : -1) + all.length) % all.length; show(all[j]);
  }}
}});
document.querySelectorAll(".cl").forEach(s => paint(s.id.slice(2))); count();
document.getElementById("alldedupe").onclick = () => {{
  document.querySelectorAll(".cl").forEach(s => {{
    const k = s.id.slice(2);
    if (Number(s.dataset.low) && !D.c[k]) {{ D.c[k] = "dedupe"; paint(k); }}
  }});
  count(); save();
}};
document.getElementById("export").onclick = () => {{
  const t = document.getElementById("json"), j = JSON.stringify(D);
  document.getElementById("out").hidden = false; t.value = j; t.select();
  (navigator.clipboard ? navigator.clipboard.writeText(j) : Promise.reject()).then(
    () => document.getElementById("status").textContent = "copied: save it to a file such as review.json",
    () => document.getElementById("status").textContent = "select the text above and copy it");
}};
</script></body></html>"""
    with open(os.path.join(out, "cluster_review.html"), "w") as f:
        f.write(page)
    print("wrote", os.path.join(out, "cluster_review.html"))
    return run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="~/Photos", help="only cluster images under this directory")
    ap.add_argument("--db", default="out/embeddings.sqlite3")
    ap.add_argument("--out", default="out/clusters")
    ap.add_argument("--top", type=int, default=500)
    args = ap.parse_args()
    build(args.root, args.db, args.out, args.top)


if __name__ == "__main__":
    main()
