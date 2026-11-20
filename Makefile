# Developer checks. Run on Linux or macOS (on Windows, inside the WSL distro).
UV ?= uv
EXPORT = $(UV) export --frozen --no-default-groups --group test --group bench \
	--no-emit-project --no-header --format requirements-txt

.PHONY: check lint types test fmt requirements requirements-check image limits

# Everything CI runs outside the image.
check: lint types test requirements-check

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

types:
	$(UV) run mypy

# Tests that need the model files skip outside the image.
test:
	$(UV) run pytest -q

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
