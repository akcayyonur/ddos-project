@echo off
call .\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --train data\nsl_kdd_train.csv --test data\nsl_kdd_test.csv
pause
