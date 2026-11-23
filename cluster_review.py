"""Cluster the remaining photos by CLIP embedding and build a cluster-level review page (largest first).

Theory of purpose: per-image classification (receipt/screenshot) proved unreliable; Simson prefers to judge
groups of similar images at once (keep all, delete all, keep the best copy, move to ~/Documents, or decide per
image).
Theory of operation: byte-identical files (same sha256) are grouped first, and only one of each group, the one
with the shortest path, takes part in similarity clustering; the rest are attached to its cluster as identical
copies, or form a cluster of their own. A k-nearest-neighbour graph (K neighbours, cosine >= START) is built on
the GPU when PyTorch is available (otherwise with numpy) from embeddings.sqlite3 (existing files only, never
inside .photoslibrary, one model only). Its connected components are split recursively at rising thresholds
(STEPS) while a component is larger than MAX_SIZE or loose (some member has cosine < COHESION to its centroid),
which prevents chaining. Clusters are then split wherever consecutive capture times are more than TIME_GAP hours
apart; photos with no capture time join the piece holding their most similar photo. Clusters of size >= 2 are
ranked by size. The page shows the first --top clusters, with a lightbox for full-size viewing (use
serve_review.py so every browser can show the originals) that never leaves the page, so marks are never lost.
Within each shown cluster, resolution_dupes.py finds lower-resolution copies and same-size re-saves, and every
copy (identical, lower-res or re-saved) points to the one copy to keep. Marks persist in localStorage, and "Copy
JSON" emits only cluster choices and per-image exceptions, stamped with a run id (a hash of clusters.csv).
apply_review.py uses clusters.csv to act on the marks. Read-only on photos.
"""

import argparse
import collections
import contextlib
import csv
import dataclasses
import hashlib
import html
import os
import sqlite3
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageOps

from photo_io import parse_time, register_heif
from resolution_dupes import find_copies


@dataclasses.dataclass
class Settings:
    """Clustering parameters. The defaults were set by inspecting one archive of about 120,000 photos."""

    k: int = 20                      # neighbours kept per photo
    start: float = 0.90              # minimum cosine for a link
    steps: tuple = (0.92, 0.94, 0.96, 0.98)  # thresholds for re-splitting large or loose components
    max_size: int = 300              # components larger than this are re-split
    cohesion: float = 0.82           # components with a member below this cosine to the centroid are re-split
    time_gap: float = 24.0           # hours between capture times that splits a cluster; 0 disables
    workers: int = 12                # processes for thumbnails and resolution checks


DEFAULTS = Settings()
K, START, STEPS, MAX_SIZE, COHESION = DEFAULTS.k, DEFAULTS.start, DEFAULTS.steps, DEFAULTS.max_size, DEFAULTS.cohesion


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def knn_graph(M, k=K, start=START, block=4096):
    """Symmetric neighbor lists from exact K-nearest-neighbor search over unit row vectors M (numpy or torch)."""
    torch = _torch()
    if torch is not None:
        if not isinstance(M, torch.Tensor):
            dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            M = torch.from_numpy(np.ascontiguousarray(M, np.float32)).to(dev)

        def top(S, kk):
            v, i = torch.topk(S, kk, dim=1)
            return v.cpu().numpy(), i.cpu().numpy()
    else:
        M = np.asarray(M, np.float32)

        def top(S, kk):
            i = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            return np.take_along_axis(S, i, 1), i
    n = M.shape[0]
    nbr = [[] for _ in range(n)]
    for s in range(0, n, block):
        v, i = top(M[s:s + block] @ M.T, min(k + 1, n))  # + 1 because each image finds itself
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


def split(comp, nbr, E, level, settings=DEFAULTS):
    c = E[comp].mean(0)
    c /= np.linalg.norm(c)
    loose = (E[comp] @ c).min() < settings.cohesion
    if len(comp) <= 1 or (len(comp) <= settings.max_size and not loose) or level >= len(settings.steps):
        return [comp]
    out = []
    for sub in components(comp, nbr, settings.steps[level]):
        out += split(sub, nbr, E, level + 1, settings)
    return out


def split_by_time(comp, taken, E, gap_hours):
    """Split a cluster wherever consecutive capture times are more than gap_hours apart.

    taken[i] is a datetime or None. Photos without a time join the piece holding their most similar photo.
    """
    timed = sorted((x for x in comp if taken[x] is not None), key=lambda x: taken[x])
    if not gap_hours or len(timed) < 2:
        return [comp]
    pieces, cur = [], [timed[0]]
    for a, b in zip(timed, timed[1:]):
        if (taken[b] - taken[a]).total_seconds() > gap_hours * 3600:
            pieces.append(cur)
            cur = []
        cur.append(b)
    pieces.append(cur)
    if len(pieces) == 1:
        return [comp]
    home = {x: p for p, piece in enumerate(pieces) for x in piece}
    for x in comp:
        if taken[x] is None:
            pieces[home[max(timed, key=lambda y: float(E[x] @ E[y]))]].append(x)
    return pieces


def identical_groups(paths, shas):
    """Map each duplicate index to its keeper (shortest path, then alphabetical) among files with equal sha256."""
    by_sha = collections.defaultdict(list)
    for i, h in enumerate(shas):
        if h:
            by_sha[h].append(i)
    dup = {}
    for g in by_sha.values():
        if len(g) > 1:
            keeper = min(g, key=lambda i: (len(paths[i]), paths[i]))
            dup.update({i: keeper for i in g if i != keeper})
    return dup


def cluster(E, paths, shas, taken, settings=DEFAULTS):
    """Return (clusters, identical) where clusters are lists of indices (size >= 2, largest first) and identical
    maps each byte-identical duplicate to its keeper. Duplicates are clustered through their keeper."""
    identical = identical_groups(paths, shas)
    reps = np.array([i for i in range(len(paths)) if i not in identical], dtype=np.int64)
    nbr = knn_graph(E[reps], settings.k, settings.start)
    Er, tr = E[reps], [taken[i] for i in reps]
    out = []
    for comp in components(range(len(reps)), nbr, settings.start):
        for c in split(comp, nbr, Er, 0, settings):
            out += split_by_time(c, tr, Er, settings.time_gap)
    members = {int(reps[x]): [int(reps[x])] for c in out for x in c}
    for d, keeper in identical.items():
        members[keeper].append(d)
    clusters = [sum((members[int(reps[x])] for x in c), []) for c in out]
    return sorted((c for c in clusters if len(c) >= 2), key=len, reverse=True), identical


def iid(path):
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def thumb(path, tdir):
    """Write a 200 px JPEG thumbnail, keyed on path and size so changed files get a new one."""
    name = hashlib.sha1(f"{path}|{os.stat(path).st_size}".encode()).hexdigest()[:16] + ".jpg"
    dest = os.path.join(tdir, name)
    if not os.path.exists(dest):
        try:
            register_heif()
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


def load(root, db_path, model=None):
    """Read the embedded images under root: (paths, E, rows); refuse a database that mixes models."""
    prefix = os.path.join(os.path.abspath(os.path.expanduser(root)), "")
    with contextlib.closing(sqlite3.connect(db_path)) as db:
        rows = [r for r in db.execute("SELECT path, emb, width, height, size, sha256, taken, model FROM images "
                                      "WHERE emb IS NOT NULL")
                if r[0].startswith(prefix) and os.path.exists(r[0]) and ".photoslibrary/" not in r[0]]
    models = collections.Counter(r[7] for r in rows)
    if len(models) > 1 or (model and models and set(models) != {model}):
        found = ", ".join(f"{m} ({n:,} photos)" for m, n in models.most_common())
        raise SystemExit(f"{prefix} holds embeddings from {found}; vectors from different models cannot be compared. "
                         f"Run embed_photos.py --model ... over {prefix} with one model first.")
    if not rows:
        raise SystemExit(f"no embedded photos under {prefix}; run embed_photos.py first")
    E = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
    return [r[0] for r in rows], E, rows


def build(root, db_path, out, top_n=500, settings=DEFAULTS, model=None):
    """Cluster the embedded images under root and write clusters.csv and cluster_review.html to out."""
    PHOTOS = os.path.join(os.path.abspath(os.path.expanduser(root)), "")
    tdir = os.path.join(out, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    paths, E, rows = load(root, db_path, model)
    n = len(paths)
    dims = {r[0]: f"{r[2]}×{r[3]}" for r in rows}
    taken = [parse_time(r[6]) for r in rows]
    clusters, identical = cluster(E, paths, [r[5] for r in rows], taken, settings)
    print(f"{n:,} images ({len(identical):,} byte-identical duplicates); {len(clusters):,} clusters of 2+ covering "
          f"{sum(map(len, clusters)):,} images; largest {[len(c) for c in clusters[:10]]}", flush=True)

    top = clusters[:top_n]
    pixels = [(r[2] or 0) * (r[3] or 0) for r in rows]
    nbytes = [r[4] for r in rows]
    found = find_copies([[x for x in c if x not in identical] for c in top], paths, E, pixels, nbytes,
                        os.path.join(out, "same_image_cache.json"), settings.workers)
    final = lambda x: found.get(x, x)
    copy_of = {x: final(k) for x, k in identical.items()} | found
    kind = {x: "identical" for x in identical} | {x: "lower-res" if pixels[found[x]] != pixels[x] else "re-save"
                                                   for x in found}
    print(f"{len(identical):,} identical, {sum(v == 'lower-res' for v in kind.values()):,} lower-resolution and "
          f"{sum(v == 're-save' for v in kind.values()):,} re-saved copies", flush=True)
    with open(os.path.join(out, "clusters.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "id", "path", "copy_of", "copy_kind"])
        for k, c in enumerate(clusters, 1):
            for x in c:
                w.writerow([k, iid(paths[x]), paths[x], iid(paths[copy_of[x]]) if x in copy_of else "", kind.get(x, "")])

    with open(os.path.join(out, "clusters.csv"), "rb") as f:
        run_id = hashlib.sha256(f.read()).hexdigest()[:12]  # ties review marks to this exact clustering
    with Pool(settings.workers) as pool:
        thumbs = dict(pool.map(_thumb, [(paths[x], tdir) for c in top for x in c], chunksize=16))
    n_copies = sum(x in copy_of for c in top for x in c)
    labels = {"identical": "identical copy", "lower-res": "lower-res copy", "re-save": "re-saved copy, smaller file"}

    def span(c):
        t = sorted(taken[x] for x in c if taken[x])
        if not t:
            return ""
        a, b = t[0].strftime("%Y-%m-%d"), t[-1].strftime("%Y-%m-%d")
        return f" · taken {a}" + (f" to {b}" if b != a else "")

    def cluster_html(k, c):
        c = sorted(c, key=lambda x: paths[x])
        cen = E[c].mean(0)
        cen /= np.linalg.norm(cen)
        coh = float((E[c] @ cen).mean())
        folders = collections.Counter(os.path.dirname(paths[x]).removeprefix(PHOTOS) for x in c).most_common(3)
        fold = "; ".join(f"{html.escape(f)} ({m})" for f, m in folders)
        n_low = sum(x in copy_of for x in c)
        shas = {rows[x][5] for x in c}
        same = len(shas) == 1 and None not in shas
        choices = [("keep", "Keep all"), ("delete", "Delete all"), ("move", "Move all to ~/Documents"), ("each", "Decide per image")]
        if n_low:
            choices.insert(1, ("dedupe", f"Keep best copy (delete {n_low} {'copy' if n_low == 1 else 'copies'})"))
        acts = "".join(f'<label><input type="radio" name="c{k}" value="{v}" data-c="{k}"{" checked" if v == "keep" else ""}> {t}</label>'
                       for v, t in choices)
        keepers = {copy_of[x] for x in c if x in copy_of}
        tag = lambda x: (f'<div class="res low">{labels[kind[x]]} ({dims[paths[x]]})</div>' if x in copy_of
                         else f'<div class="res high">best copy ({dims[paths[x]]})</div>' if x in keepers else "")
        imgs = "".join(
            f'<figure class="im" data-id="{iid(paths[x])}" data-c="{k}"{" data-lower=1" if x in copy_of else ""}><img loading="lazy" src="{thumbs.get(paths[x], "")}" '
            f'data-full="file://{html.escape(paths[x])}" data-path="{html.escape(paths[x].removeprefix(PHOTOS))}" data-dims="{dims[paths[x]]}">'
            f'{tag(x)}<figcaption>{html.escape(os.path.basename(paths[x]))}</figcaption>'
            f'<select data-id="{iid(paths[x])}" data-c="{k}"><option value="">(cluster choice)</option><option value="keep">keep</option>'
            f'<option value="delete">delete</option><option value="move">move to Documents</option></select></figure>' for x in c)
        what = "identical files" if same else f"{n_low} {'copy' if n_low == 1 else 'copies'}" if n_low else ""
        return (f'<section class="cl{" identical" if same else ""}" id="cl{k}" data-low="{n_low}"><h2>#{k} · {len(c)} images{f" · {what}" if what else ""}{span(c)} · cohesion {coh:.3f}</h2><div class="m">{fold}</div>'
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
<span id="count"></span><button id="alldedupe" title="Applies to clusters still set to Keep all">Keep best copy in all clusters ({n_copies:,} copies)</button><button id="export">Copy JSON</button><span id="status"></span></div>
<div id="out" hidden><textarea id="json" readonly rows="4"></textarea></div>
<p><b>Keep best copy</b> deletes only byte-identical files and files confirmed to be the same uncropped picture as a better
file in the cluster: one with more pixels, or the same pixels in a larger, less compressed file.
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


def add_cluster_args(ap):
    """Clustering options shared by cluster_review.py and photo_clusters.py; returns a function making Settings."""
    d = DEFAULTS
    g = ap.add_argument_group("clustering")
    g.add_argument("--k", type=int, default=d.k, help=f"neighbours kept per photo (default {d.k})")
    g.add_argument("--start", type=float, default=d.start, help=f"minimum cosine for a link (default {d.start})")
    g.add_argument("--steps", default=",".join(map(str, d.steps)),
                   help=f"comma-separated thresholds for re-splitting (default {','.join(map(str, d.steps))})")
    g.add_argument("--max-size", type=int, default=d.max_size, help=f"re-split larger components (default {d.max_size})")
    g.add_argument("--cohesion", type=float, default=d.cohesion,
                   help=f"re-split components with a member below this cosine to the centroid (default {d.cohesion})")
    g.add_argument("--time-gap", type=float, default=d.time_gap,
                   help=f"split clusters at capture-time gaps over this many hours; 0 disables (default {d.time_gap:g})")
    g.add_argument("--cluster-workers", type=int, default=d.workers,
                   help=f"processes for thumbnails and resolution checks (default {d.workers})")
    return lambda a: Settings(a.k, a.start, tuple(float(x) for x in a.steps.split(",") if x.strip()), a.max_size,
                              a.cohesion, a.time_gap, a.cluster_workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="~/Photos", help="only cluster images under this directory")
    ap.add_argument("--db", default="out/embeddings.sqlite3")
    ap.add_argument("--out", default="out/clusters")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--model", help="refuse unless every photo was embedded with this model id, e.g. ViT-B-32/laion2b_s34b_b79k")
    settings = add_cluster_args(ap)
    args = ap.parse_args()
    build(args.root, args.db, args.out, args.top, settings(args), args.model)


if __name__ == "__main__":
    main()
