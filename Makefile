install_precommit:
	pip install black flake8 isort mypy pre-commit
	pre-commit install --hook-type pre-commit

install_deps:
	pip install -r requirements.txt

download_dataset:
	. ./scripts/download_dataset.sh
	. ./scripts/move_data.sh

install: install_precommit install_deps
