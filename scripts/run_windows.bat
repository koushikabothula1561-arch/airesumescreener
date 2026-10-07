@echo off
call .venv\Scripts\activate
python src\train.py
uvicorn app.main:app --reload
