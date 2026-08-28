.PHONY: help install verify test check

help:
	@echo "BrainAxL development commands"
	@echo "  make install  Install the framework with test dependencies"
	@echo "  make verify   Validate submission provenance and repository hygiene"
	@echo "  make test     Run the lightweight BrainAxL architecture tests"
	@echo "  make check    Run verify and test"

install:
	python -m pip install -e 'asparagus[test]'

verify:
	python scripts/verify_release.py

test:
	PYTHONPATH=asparagus python -m pytest asparagus/tests/test_brainaxl.py

check: verify test
