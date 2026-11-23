# sample writes a small synthetic photo directory (sample/) for a first try.
# Build the environment, embed a photo tree, build the cluster review page, and run tests.
# ROOT is the photo directory (never modified); DB and OUT hold private local outputs.
# TOP is the number of clusters shown on the page, largest first.
# Embeddings are incremental, so rerunning embed only processes new or changed files.
# review needs embed to have run first; test needs neither. run does embed + review in one step.
# apply is a dry run unless EXECUTE=--execute; JSON is a file holding the Copy JSON output.
#
# Optional settings; each is passed on only when set, so the programs' own defaults apply otherwise
# (see `uv run python photo_clusters.py --help`):
#   embedding:  MODEL, PRETRAINED, BATCH, WORKERS
#   clustering: K, START, STEPS (comma-separated), MAX_SIZE, COHESION, TIME_GAP (hours; 0 disables)
# Changing MODEL or PRETRAINED re-embeds every photo; set both, since weights belong to one model
# (e.g. MODEL=ViT-L-14 PRETRAINED=laion2b_s32b_b82k). Example: make run ROOT=~/Pictures START=0.88 TIME_GAP=12
#
# test runs every test, including the end-to-end test in a headless browser (Playwright's Chromium is
# downloaded once, about 150 MB). e2e-clip runs the end-to-end test with the real CLIP model instead of the
# stand-in embedder (downloads the model, about 600 MB, the first time).

ROOT ?= sample
DB ?= out/embeddings.sqlite3
OUT ?= out/clusters
TOP ?= 500

opt = $(if $($(1)),--$(2) "$($(1))")
EMBED_ARGS = $(call opt,MODEL,model) $(call opt,PRETRAINED,pretrained) $(call opt,BATCH,batch) $(call opt,WORKERS,workers)
CLUSTER_ARGS = $(call opt,K,k) $(call opt,START,start) $(call opt,STEPS,steps) $(call opt,MAX_SIZE,max-size) \
               $(call opt,COHESION,cohesion) $(call opt,TIME_GAP,time-gap)
PLAYWRIGHT = uv run --with playwright

.PHONY: sync sample run serve apply embed review test e2e e2e-clip browser
sync:
	uv sync

sample:
	uv run python make_sample.py sample

run:
	uv run python photo_clusters.py "$(ROOT)" "$(DB)" --out "$(OUT)" --top $(TOP) $(EMBED_ARGS) $(CLUSTER_ARGS)

serve:
	uv run python serve_review.py --out "$(OUT)"

apply:
	uv run python apply_review.py "$(JSON)" "$(ROOT)" "$(DB)" --clusters "$(OUT)/clusters.csv" $(EXECUTE)

embed:
	uv run python embed_photos.py "$(ROOT)" --db "$(DB)" $(EMBED_ARGS)

review:
	uv run python cluster_review.py --root "$(ROOT)" --db "$(DB)" --out "$(OUT)" --top $(TOP) $(CLUSTER_ARGS)

browser:
	$(PLAYWRIGHT) python -m playwright install chromium

test: browser
	$(PLAYWRIGHT) python -m unittest discover -s tests -v

e2e: browser
	$(PLAYWRIGHT) python -m unittest tests.test_end_to_end -v

e2e-clip: browser
	E2E_MODEL=clip $(PLAYWRIGHT) python -m unittest tests.test_end_to_end -v
