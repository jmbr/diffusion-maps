#############################################################################
# Development make file.
#############################################################################

SOURCE_DIR := diffusion_maps

$(SOURCE_DIR)/version.py:
	@scripts/make-version > $@

conda: conda-recipe/meta.yaml
	conda build conda-recipe

conda-recipe/meta.yaml: conda-recipe/meta.yaml.in $(SOURCE_DIR)/version.py
	@VERSION=`grep v_short diffusion_maps/version.py | cut -f 2 -d "'"`; \
	sed "s/VERSION/$$VERSION/g" conda-recipe/meta.yaml.in > $@

.PHONY: $(SOURCE_DIR)/version.py conda
