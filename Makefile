install_precommit:
	pip install black flake8 isort mypy pre-commit
	pre-commit install --hook-type pre-commit

download_dataset:
	. ./scripts/download_dataset.sh

install: install_precommit
