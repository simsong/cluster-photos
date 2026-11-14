# sample writes a small synthetic photo directory (sample/) for a first try.
# Build the environment, embed a photo tree, build the cluster review page, and run tests.
# ROOT is the photo directory (never modified); DB and OUT hold private local outputs.
# TOP is the number of clusters shown on the page, largest first.
# Embeddings are incremental, so rerunning embed only processes new or changed files.
# review needs embed to have run first; test needs neither. run does embed + review in one step.
# apply is a dry run unless EXECUTE=--execute; JSON is a file holding the Copy JSON output.

ROOT ?= sample
DB ?= out/embeddings.sqlite3
OUT ?= out/clusters
TOP ?= 500

.PHONY: sync sample run serve apply embed review test
sync:
	uv sync

sample:
	uv run python make_sample.py sample

run:
	uv run python photo_clusters.py "$(ROOT)" "$(DB)" --out "$(OUT)" --top $(TOP)

serve:
	uv run python serve_review.py --out "$(OUT)"

apply:
	uv run python apply_review.py "$(JSON)" "$(ROOT)" "$(DB)" --clusters "$(OUT)/clusters.csv" $(EXECUTE)

embed:
	uv run python embed_photos.py "$(ROOT)" --db "$(DB)"

review:
	uv run python cluster_review.py --root "$(ROOT)" --db "$(DB)" --out "$(OUT)" --top $(TOP)

test:
	uv run python -m unittest discover -s tests -v
