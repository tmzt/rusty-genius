.PHONY: ogenius_tests

test_binary := $(shell pwd)/target/release/ogenius-metal

ogenius_metal:
	cargo build --release -p ogenius --features metal
	cp target/release/ogenius $(test_binary)

ogenius_tests: ogenius_metal
	TEST_BINARY=$(test_binary) cargo test --package ogenius --test http_tests -- --test-threads=1

ogenius_embed: ogenius_metal
	TEST_BINARY=$(test_binary) cargo run --package ogenius --bin embed_test
