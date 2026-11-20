# Developer checks and the evidence pipeline. Run on Linux or macOS (on Windows, inside the WSL distro).
UV ?= uv
EXPORT = $(UV) export --frozen --no-default-groups --group test --group bench \
	--no-emit-project --no-header --format requirements-txt

# The results root. The runner passes it to compose as an absolute path, and the compose files
# mount it at /app/results.
RRI_RESULTS ?= $(CURDIR)/results
REPEATS ?= 3
# real, fake or null.
MODELS ?= real
# Where the rendered DESIGN.md and README.md are written.
RENDER_OUT ?= $(CURDIR)
# current checks the evidence against the code as it is now; recorded takes the runs facts.json
# names.
DIGEST ?= current

.PHONY: check lint types test test-docker fmt requirements requirements-check image limits \
	evidence compare report numbers

# Everything CI runs outside the image. The number check takes the runs facts.json recorded,
# so a code change alone does not fail it on committed evidence.
check: lint types test requirements-check
	$(UV) run python -m bench.numbers --results results --out . --keep-missing --digest recorded

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

types:
	$(UV) run mypy

# Tests that need the model files skip outside the image. Tests that need the host's docker CLI
# are marked docker and run in test-docker.
test:
	$(UV) run pytest -q -m "not docker"

# On the host, with the docker CLI.
test-docker:
	$(UV) run pytest -q -m docker

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# requirements.txt is what the image installs. It is exported from uv.lock.
requirements:
	$(EXPORT) -o requirements.txt

requirements-check:
	$(EXPORT) | diff - requirements.txt

image:
	docker compose build

# Print the CPU and memory limits as read inside the container.
limits:
	docker compose run --rm system python -m common.cgroup

# On the host: every scenario of the plan, gated, in fresh containers. Finished runs are skipped.
evidence:
	$(UV) run python -m bench.evidence run --results $(RRI_RESULTS) --repeats $(REPEATS) --models $(MODELS)

# On the host: every comparison, on the real models whatever MODELS is.
compare:
	$(UV) run python -m bench.evidence compare all --results $(RRI_RESULTS)

# Each line stops the target when it fails, so nothing is rendered from a refused build. The
# last line runs after everything is written; it exits 1 for a failed check, and on the real
# models for a key the templates use that is null.
report:
	$(UV) run python -m bench.summarize --results $(RRI_RESULTS)
	$(UV) run python -m bench.aggregate build --results $(RRI_RESULTS) --models $(MODELS) --digest $(DIGEST)
	$(UV) run python -m bench.tables --results $(RRI_RESULTS)
	$(UV) run python -m bench.metrics_table --results $(RRI_RESULTS)
	$(UV) run python -m bench.plots --results $(RRI_RESULTS)
	$(UV) run python -m bench.render --results $(RRI_RESULTS) --out $(RENDER_OUT)
	$(UV) run python -m bench.numbers --results $(RRI_RESULTS) --out $(RENDER_OUT) --models $(MODELS) --digest $(DIGEST)
	$(UV) run python -m bench.aggregate status --results $(RRI_RESULTS)

numbers:
	$(UV) run python -m bench.numbers --results $(RRI_RESULTS) --out $(RENDER_OUT) --models $(MODELS) --digest $(DIGEST)
