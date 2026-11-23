# cluster-photos

Find groups of visually similar photos in a large collection, review them in your browser, and clean them up.

cluster-photos walks a directory tree of photos, computes a [CLIP](https://github.com/mlfoundations/open_clip)
embedding for every image, and groups images that look alike: byte-identical files, re-saved and resized copies,
bursts, and photos of the same scene taken around the same time. It writes a review page that shows each group
side by side, so you can decide a whole group at once ("keep all", "delete all", "keep only the best copy", or
per photo). A separate
program then carries out your decisions, as a dry run unless you ask it to act.

It was built for an archive of about 120,000 photos. No labels, no cloud service and no account are needed;
everything runs on your own computer, and your photos are only read until you explicitly apply a review.

## Requirements

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS, or see the uv docs).
- About 2 GB of disk for PyTorch and the model. The CLIP weights (about 600 MB) are downloaded from Hugging
  Face the first time you run it.
- A GPU helps but is not required. Apple Silicon (MPS) and NVIDIA (CUDA) are used automatically; otherwise it
  runs on the CPU.
- Embedding, clustering and the review page work on macOS and Linux. Applying deletions uses the macOS
  `/usr/bin/trash` command, so `apply_review.py --execute` is macOS-only (deleted photos go to the Trash).

## Quick start

```bash
git clone https://github.com/simsong/cluster-photos.git
cd cluster-photos
make sync          # create .venv and install dependencies with uv
make test          # optional: unit tests plus an end-to-end test in a headless browser (no model or photos needed)
```

Try it first on a small synthetic sample (33 generated images in `sample/`):

```bash
make sample
make run           # embeds sample/, clusters it, writes out/clusters/cluster_review.html
make serve         # then open http://127.0.0.1:8765/
```

Then run it on your own photos. `ROOT` is searched recursively; hidden folders and files are skipped.

```bash
make run ROOT=~/Pictures
make serve
```

Open <http://127.0.0.1:8765/> in your browser. Press Ctrl-C in the terminal to stop the server.

All outputs go under `out/` (ignored by git): `out/embeddings.sqlite3` holds the embeddings and
`out/clusters/` holds `clusters.csv`, thumbnails and the review page. You can change these with
`DB=...` and `OUT=...`, and change how many clusters the page shows with `TOP=...` (default 500, largest first).

Reruns are incremental: only new or changed files are embedded again, and rows for deleted files are dropped,
so you can run `make run` again after each round of cleanup.

### Settings

Every setting has a command-line flag (`--help` lists them with their defaults) and a `make` variable that is
passed on only when you set it:

| `make` variable | Flag | Default | Meaning |
|---|---|---|---|
| `MODEL`, `PRETRAINED` | `--model`, `--pretrained` | `ViT-B-32`, `laion2b_s34b_b79k` | open_clip model and weights |
| `BATCH`, `WORKERS` | `--batch`, `--workers` | 128, 12 | embedding batch size and file-reading processes |
| `K` | `--k` | 20 | neighbors kept per photo |
| `START` | `--start` | 0.90 | minimum cosine similarity for two photos to be linked |
| `STEPS` | `--steps` | `0.92,0.94,0.96,0.98` | stricter thresholds for re-splitting large or loose groups |
| `MAX_SIZE` | `--max-size` | 300 | groups larger than this are re-split |
| `COHESION` | `--cohesion` | 0.82 | groups with a member less similar than this to the group average are re-split |
| `TIME_GAP` | `--time-gap` | 24 | split groups where capture times are more than this many hours apart; 0 turns it off |

For example, `make run ROOT=~/Pictures START=0.88 TIME_GAP=12`. Changing the clustering settings only re-runs
the clustering, which takes about a minute. Changing `MODEL` or `PRETRAINED` re-embeds every photo, because
vectors from different models cannot be compared; set both, since weights belong to one model (for example
`MODEL=ViT-L-14 PRETRAINED=laion2b_s32b_b82k`). Clustering refuses a database that mixes models.

### Why a local web server?

The review page is a static HTML file, but browsers restrict what a `file://` page may load (Safari will not show
the full-size originals), and Chrome cannot display HEIC at all. `serve_review.py` serves the page and the
originals on `127.0.0.1` only, converts HEIC and TIFF to JPEG in memory, and serves only photos that appear in
`clusters.csv`. It never writes to your photos. Opening the HTML file directly still works for thumbnails.

## Reviewing and applying decisions

On the review page every cluster starts as **Keep all**, so nothing happens unless you choose. For each cluster
you can pick Keep all, Keep best copy (only when copies were found), Delete all, Move all, or Decide per image.
A copy is a byte-identical file, a lower-resolution version, or a same-size re-save in a smaller file; each is
labelled on the page, and the copy that would be kept is marked "best copy". Clusters made only of
byte-identical files say "identical files" in their heading. Click a photo to see it full size; use the arrow keys to step through its
cluster and Esc to close. Your marks are saved in the browser as you go.

When you are done, press **Copy JSON** and save the text to a file, for example `review.json`. Then do a dry
run, which only prints what would happen:

```bash
make apply ROOT=~/Pictures JSON=review.json
```

If that looks right, carry it out:

```bash
make apply ROOT=~/Pictures JSON=review.json EXECUTE=--execute
```

Deleted photos go to the macOS Trash, moved photos go to `~/Documents/From Photos` keeping their subfolders
(change this with `apply_review.py --dest`), nothing is ever overwritten, and every action is logged to
`out/applied.csv`. A file is skipped if it changed since it was scanned. Marks from an older clustering run are
refused, because cluster numbers change between runs, so apply a review before running `make run` again.

## Programs

| Program | Input | Output |
|---|---|---|
| `photo_clusters.py` | a directory and a database path | updated embeddings, `clusters.csv`, and the review page |
| `embed_photos.py` | a directory of images | one vector per image (512 numbers for the default model), its SHA-256 and capture time, in SQLite |
| `cluster_review.py` | that SQLite database | `clusters.csv` and a static HTML review page |
| `serve_review.py` | the output directory | the review page and full-size photos on 127.0.0.1 |
| `apply_review.py` | the page's Copy JSON output | photos moved to the Trash or to another folder |
| `rename_review.py`, `apply_renames.py` | a CSV of proposed new names | a review page for renames, then reviewed renames with an undo log |
| `make_sample.py` | nothing | a small synthetic photo directory for trying the pipeline |

Only `apply_review.py` and `apply_renames.py` change files, and only with `--execute`. Every program takes
`--help`.

Supported formats are JPEG, HEIC, PNG, GIF, TIFF, WebP and BMP. Files inside macOS `.photoslibrary` bundles are
never clustered or changed: export from Photos first if you want to review those.

## How it works

### 1. Embedding (`embed_photos.py`)

Each image goes through CLIP (Radford et al., 2021), a vision transformer trained so that images and their
captions land near each other. The model is `open_clip` ViT-B/32 with the `laion2b_s34b_b79k` weights.

- A `DataLoader` with worker processes reads each file once, computes its SHA-256, and decodes it. Decoding
  applies the EXIF orientation, reads HEIC through `pillow-heif`, and uses JPEG DCT scaling (`draft`) for speed.
- `open_clip`'s preprocessing resizes the short side to 224 pixels and takes the center crop.
- The output is L2-normalized, so the dot product of two vectors is their cosine similarity.
- The EXIF capture time (`DateTimeOriginal`) and the file's SHA-256 are stored with each vector.
- Vectors are stored as raw float32 blobs, keyed by path, with the model that made them. A file is skipped if
  its size, mtime and model are unchanged, so reruns are incremental; changing the model re-embeds everything.
- `--model tiny-thumbnail` is a stand-in used by the tests: a 16 × 16 color thumbnail instead of CLIP, with no
  PyTorch and no download. It finds copies of one picture but not merely similar scenes.

On an Apple M3 Max using the MPS GPU, 126,824 images took about 16 minutes (135 images per second on average).

### 2. Byte-identical files

Files with the same SHA-256 are grouped first. Only one of each group, the one with the shortest path (then
alphabetical), takes part in the similarity clustering, so a dozen copies of one file cannot fill a photo's
neighbor list. The others join that photo's cluster as identical copies, or form a cluster of their own if it
resembles nothing else.

### 3. Nearest-neighbor graph (`cluster_review.py`)

For every image, the program finds its `K = 20` nearest neighbors by exact search: blocks of 4,096 rows are
multiplied against the whole matrix on the GPU, and `torch.topk` keeps the best of each row (numpy is used if
PyTorch is not installed). This is still
O(n²d) work, but about 7 billion comparisons take seconds because the arithmetic runs as one dense matrix
multiply. No approximate index is used, so no neighbor is missed.

Links with cosine ≥ `START = 0.90` become undirected edges of a sparse graph.

### 4. Clustering by connected components with recursive splitting

The connected components of a graph thresholded at similarity *t* are exactly the clusters of single-linkage
hierarchical clustering cut at *t* (Gower & Ross, 1969), here computed on the K-nearest-neighbor graph.
`components` finds them with an iterative depth-first search, in time linear in the graph size.

Single linkage alone suffers from *chaining*: A resembles B and B resembles C, so a long path can join photos
that have nothing in common. `split` guards against that. It re-cuts a component at the next threshold in
`STEPS = (0.92, 0.94, 0.96, 0.98)` when either of these is true:

- it has more than `MAX_SIZE = 300` images, or
- it is loose: some member has cosine below `COHESION = 0.82` to the normalized centroid.

The result is a coarse divisive hierarchy. Tight groups stop early and stay whole, while loose ones are broken
along their weakest links. No cluster count is chosen in advance, and images with no close neighbor are simply
left out. A component still large or loose after 0.98 is kept as it is.

Finally each cluster is split wherever two consecutive capture times are more than `TIME_GAP = 24` hours apart,
so the same view photographed on two trips becomes two clusters. Photos with no capture time (scans,
screenshots, many PNGs) join the piece holding their most similar photo, and a cluster with fewer than two
timed photos is left whole.

Clusters of two or more images are ranked by size. On a 116,148-image archive, before identical files and capture
times were taken into account, this produced 16,174 clusters covering 44,296 photos, with the largest holding
266, 166 and 133. This step takes about a minute.

### 5. Same image at a different resolution or re-saved (`resolution_dupes.py`)

Inside each shown cluster, a pair is treated as one image stored twice only if its pixel counts differ by at
least 5% (a lower-resolution copy) or are equal with different file sizes (a re-save), CLIP cosine is at least
0.97, and it passes registration: SIFT matches and a RANSAC similarity transform, at
least 85% mutual overlap (so a crop never counts) and aligned correlation of at least 0.95. Pairs with different
camera frame numbers or capture times over 2 s apart are rejected. Files are visited best first (more pixels,
then more bytes), and a file is marked as a copy only if it directly matches a better file that is itself kept.
For a re-save, the larger file is kept on the reasoning that it is less compressed; this compares bytes across
formats too, so a PNG and a JPEG of the same picture keep whichever file is larger. Byte-identical copies point
to the same keeper as the file they duplicate. The page offers *Keep best copy* per cluster and for all clusters.

### 6. Applying decisions (`apply_review.py`)

The JSON is validated with a Pydantic model and resolved against `clusters.csv`: a photo's own mark overrides
its cluster's action, no mark means keep, and *dedupe* deletes only listed copies whose keeper stays.
Each page carries a run id (a hash of `clusters.csv`), and marks from a different clustering are refused.

The default is a dry run. With `--execute`, each file must be under the root, outside any `.photoslibrary`, and
still match its scanned SHA-256.

## Limits

- CLIP measures *visual and semantic* similarity. A cluster means "these look alike", which covers duplicates,
  bursts and simply similar scenes. It does not prove that two files are the same photo, so review before
  deleting.
- The center crop and the embedding are not rotation-invariant, so a photo stored sideways can land elsewhere.
- Only each image's K strongest links are considered, so a very large set of near-identical images is linked
  through those neighbors rather than all-to-all.
- The default `START`, `STEPS`, `MAX_SIZE`, `COHESION` and `TIME_GAP` were set by inspecting one archive, and were
  not tuned against labeled data. Each can be changed (see Settings).
- Capture times come from the camera's clock in local time, with no time zone, so a wrong camera clock can split
  or join clusters by mistake.
- The whole embedding matrix is held in memory (about 2 KB per image), which is fine for a few hundred thousand
  photos.
- `out/` contains your photo paths, thumbnails and embeddings. It is ignored by git; do not share it.

## Tests

`make test` runs the unit tests and an end-to-end test. The end-to-end test generates the sample photos, runs
`photo_clusters.py` with the stand-in embedder, opens the review page through `serve_review.py` in a headless
Chromium (via Playwright, installed on first use), checks the clusters and labels, opens a full-size photo,
chooses *Keep best copy in all clusters*, adds a per-photo exception, presses *Copy JSON*, runs
`apply_review.py` as a dry run, carries the plan out against a temporary trash folder, and runs the pipeline
again to confirm nothing is re-embedded and no copies remain. `make e2e-clip` runs the same test with the real
CLIP model. GitHub Actions runs both on macOS for every pull request.

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
- Ilharco et al., *OpenCLIP*, 2021. Schuhmann et al., *LAION-5B*, NeurIPS 2022.
- Gower & Ross, *Minimum Spanning Trees and Single Linkage Cluster Analysis*, Applied Statistics, 1969.

## License

MIT. See [LICENSE](LICENSE).
