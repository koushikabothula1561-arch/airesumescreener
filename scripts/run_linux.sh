#!/bin/bash
source .venv/bin/activate
python src/train.py
uvicorn app.main:app --reload
