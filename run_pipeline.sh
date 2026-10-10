#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
python -m src.data
python -m src.train
python -m src.evaluate
python -m src.register
python -m src.predict
python -m src.reproduce
