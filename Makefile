.PHONY: ogenius_tests ogenius_metal

# Use local tmp for sandboxed builds
export TMPDIR := $(shell pwd)/target/tmp
export CARGO_INCREMENTAL := 0
export TARGET_DIR := $(shell pwd)/target/release

test_binary := $(TARGET_DIR)/ogenius-metal

ogenius_metal:
	mkdir -p $(TMPDIR)
	cargo build --release -p ogenius --features metal
	cp target/release/ogenius $(test_binary)

run_ogenius:
	$(test_binary) serve --addr 127.0.0.1:9099 --no-open -- --load-models "tiny-model" --load-models "embedding-gemma"

ogenius_tests: ogenius_metal
	TEST_BINARY=$(test_binary) cargo test --package ogenius --test http_tests -- --test-threads=1

ogenius_embed: ogenius_metal
	TEST_BINARY=$(TARGET_DIR)/ogenius-metal cargo run --package ogenius --bin embed_test -- --load-models "embedding-gemma"

ogenius_inference: ogenius_metal
	TEST_BINARY=$(TARGET_DIR)/ogenius-metal cargo run --package ogenius --bin inference_test -- --load-models "tiny-model"
