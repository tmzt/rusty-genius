.PHONY: ogenius_tests

ogenius_tests:
	cargo build --release --bin ogenius --features metal
	cp target/release/ogenius target/release/ogenius-metal
	TEST_BINARY=target/release/ogenius-metal cargo test --package ogenius --test http_tests -- --test-threads=1
