# cluster-photos

Find groups of visually similar photos in a large collection, review them in your browser, and clean them up.

cluster-photos walks a directory tree of photos, computes a [CLIP](https://github.com/mlfoundations/open_clip)
embedding for every image, and groups images that look alike: duplicates, re-saved and resized copies, bursts,
and photos of the same scene. It writes a review page that shows each group side by side, so you can decide a
whole group at once ("keep all", "delete all", "keep only the highest-resolution copy", or per photo). A separate
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
make test          # optional: 33 fast tests, no model or photos needed
```

Try it first on a small synthetic sample (29 generated images in `sample/`):

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

### Why a local web server?

The review page is a static HTML file, but browsers restrict what a `file://` page may load (Safari will not show
the full-size originals), and Chrome cannot display HEIC at all. `serve_review.py` serves the page and the
originals on `127.0.0.1` only, converts HEIC and TIFF to JPEG in memory, and serves only photos that appear in
`clusters.csv`. It never writes to your photos. Opening the HTML file directly still works for thumbnails.

## Reviewing and applying decisions

On the review page every cluster starts as **Keep all**, so nothing happens unless you choose. For each cluster
you can pick Keep all, Keep highest resolution (only when lower-resolution copies were found), Delete all,
Move all, or Decide per image. Click a photo to see it full size; use the arrow keys to step through its
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
| `embed_photos.py` | a directory of images | one 512-number vector per image, in SQLite |
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
- Vectors are stored as raw float32 blobs, keyed by path. A file is skipped if its size and mtime are unchanged,
  so reruns are incremental.

On an Apple M3 Max using the MPS GPU, 126,824 images took about 16 minutes (135 images per second on average).

### 2. Nearest-neighbor graph (`cluster_review.py`)

For every image, the program finds its `K = 20` nearest neighbors by exact search: blocks of 4,096 rows are
multiplied against the whole matrix on the GPU, and `torch.topk` keeps the best of each row. This is still
O(n²d) work, but about 7 billion comparisons take seconds because the arithmetic runs as one dense matrix
multiply. No approximate index is used, so no neighbor is missed.

Links with cosine ≥ `START = 0.90` become undirected edges of a sparse graph.

### 3. Clustering by connected components with recursive splitting

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

Clusters of two or more images are ranked by size. On a 116,148-image archive this produced 16,174 clusters
covering 44,296 photos, with the largest holding 266, 166 and 133. This step takes about a minute.

### 4. Same image at different resolutions (`resolution_dupes.py`)

Inside each shown cluster, a pair is treated as one image at two sizes only if its pixel counts differ by at least
5%, CLIP cosine is at least 0.97, and it passes registration: SIFT matches and a RANSAC similarity transform, at
least 85% mutual overlap (so a crop never counts) and aligned correlation of at least 0.95. Pairs with different
camera frame numbers or capture times over 2 s apart are rejected. Files are visited largest first, and a file
is marked lower-res only if it directly matches a larger file that is itself kept. The page offers *Keep highest
resolution* per cluster and for all clusters.

### 5. Applying decisions (`apply_review.py`)

The JSON is validated with a Pydantic model and resolved against `clusters.csv`: a photo's own mark overrides
its cluster's action, no mark means keep, and *dedupe* deletes only listed lower-res copies whose keeper stays.
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
- `START`, `STEPS`, `MAX_SIZE` and `COHESION` were set by inspecting one archive, and were not tuned against
  labeled data.
- The whole embedding matrix is held in memory (about 2 KB per image), which is fine for a few hundred thousand
  photos.
- `out/` contains your photo paths, thumbnails and embeddings. It is ignored by git; do not share it.

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
- Ilharco et al., *OpenCLIP*, 2021. Schuhmann et al., *LAION-5B*, NeurIPS 2022.
- Gower & Ross, *Minimum Spanning Trees and Single Linkage Cluster Analysis*, Applied Statistics, 1969.

## License

MIT. See [LICENSE](LICENSE).
