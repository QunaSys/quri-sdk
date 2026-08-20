# Local docs build for the QURI SDK Sphinx site.
# Run `make help` for targets.

SOURCEDIR = source
BUILDDIR  = _build/html
UV_RUN    = uv run
SDK_DOCS = ../quri-parts/docs

.PHONY: help install html open serve api clean

help:
	@echo "make install  - set up the docs env (one-time)"
	@echo "make html     - build the site into $(BUILDDIR)"
	@echo "make open     - build, then open in the browser"
	@echo "make serve    - live-reload server (rebuilds on save)"
	@echo "make api      - build + stage the quri-parts API ref under /api/ (slow)"
	@echo "make clean     - remove build output"

install:
	uv sync

html:
	$(UV_RUN) sphinx-build -b html $(SOURCEDIR) $(BUILDDIR)

open: html
	open $(BUILDDIR)/index.html

serve:
	uv run --with sphinx-autobuild sphinx-autobuild $(SOURCEDIR) $(BUILDDIR) --open-browser --ignore "$(BUILDDIR)/*"

# Mirrors the CI step; needs the root SDK environment, Julia, and pandoc.
api:
	@set -e; \
		docs=$(SDK_DOCS); \
		staged=$(SOURCEDIR)/_api_staged/api; \
		index="$$docs/index.rst"; \
		reference="$$docs/reference.rst"; \
		backup=$$(mktemp -d); \
		cp "$$index" "$$backup/index.rst"; \
		cp "$$reference" "$$backup/reference.rst"; \
		trap 'cp "$$backup/index.rst" "$$index"; cp "$$backup/reference.rst" "$$reference"; rm -rf "$$backup" "$$docs/_build" "$$docs/quri_parts"' EXIT; \
		printf '%s\n' 'API Reference' '========' '' '.. toctree::' '  :maxdepth: 1' '' '  reference' > "$$index"; \
		tail -n +4 "$$reference" > "$$backup/reference.build"; \
		cp "$$backup/reference.build" "$$reference"; \
		$(UV_RUN) python scripts/sort_api_toctree.py "$$reference"; \
		rm -rf "$$docs/_build" "$$docs/quri_parts"; \
		(cd .. && uv run --no-sync $(MAKE) -C quri-parts/docs html); \
		rm -rf "$$staged"; \
		mkdir -p "$$staged"; \
		cp -R "$$docs/_build/." "$$staged/"
	$(UV_RUN) python scripts/brand_api_ref.py $(SOURCEDIR)/_api_staged/api
	rm -rf $(BUILDDIR)
	$(MAKE) html

clean:
	rm -rf _build $(SOURCEDIR)/_api_staged $(SDK_DOCS)/_build $(SDK_DOCS)/quri_parts
