TARGET_FILES := $$(git diff --name-only main quri_sdk_notebooks/ | grep .ipynb)
TARGET_FILES_ALL := $$(find quri_sdk_notebooks -mindepth 1 -type f -name "*.ipynb")
EXCLUDE_FILES := .exclude
FILTER_STRING := (.cells[] | select(has("execution_count")) | .execution_count) = null | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}} | .cells[].metadata = {}


execute-in-place:
	for f in $(TARGET_FILES); do \
		if grep -Fxq $${f} $(EXCLUDE_FILES) ; then \
			echo "Skipping execution of $${f}" ; \
		else \
			echo "Executing $${f}" ; \
			poetry run jupyter execute --inplace --JupyterApp.log_level=50 $${f} ; \
		fi \
	done


execute-in-place-all:
	for f in $(TARGET_FILES_ALL); do \
		if grep -Fxq $${f} $(EXCLUDE_FILES) ; then \
			echo "Skipping execution of $${f}" ; \
		else \
			echo "Executing $${f}" ; \
			poetry run jupyter execute --inplace --JupyterApp.log_level=50 $${f} ; \
		fi \
	done


clean-nb:
	for f in $(TARGET_FILES_ALL); do \
		jq '$(FILTER_STRING)' "$${f}" > "$${f}.tmp" ; \
		mv "$${f}.tmp" "$${f}" ; \
	done
