# Directories
INPUT_DIR = oct_1_period
OUTPUT_DIR = oct_1_newline

# Python environment
PYTHON = python

# Script to split sentences
SPLIT_SCRIPT = split_sentences.py
COUNT_SCRIPT = count_sentences.py

# List of input files
INPUT_FILES = $(wildcard $(INPUT_DIR)/*.txt)

# List of output files
OUTPUT_FILES = $(patsubst $(INPUT_DIR)/%, $(OUTPUT_DIR)/%, $(INPUT_FILES))

# Default target
all: $(OUTPUT_FILES)

# Rule to create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Rule to process each file
$(OUTPUT_DIR)/%.txt: $(INPUT_DIR)/%.txt | $(OUTPUT_DIR)
	$(PYTHON) $(SPLIT_SCRIPT) -i $< -o $@

# PHONY rule to count sentences in all newline files
.PHONY: count_sentences
count_sentences:
	@for file in $(wildcard $(OUTPUT_DIR)/*.txt); do \
		echo "Processing $$file..."; \
		$(PYTHON) $(COUNT_SCRIPT) $$file; \
	done


# Clean up
clean:
	rm -rf $(OUTPUT_DIR)
