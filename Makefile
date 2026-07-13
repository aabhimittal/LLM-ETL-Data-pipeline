.PHONY: install demo test serve clean

install:
	pip install -r requirements.txt

demo:
	python examples/run_demo.py

test:
	python -m pytest

serve:
	uvicorn serve:app --reload

clean:
	rm -rf .rule_cache .pytest_cache data/customers_clean.csv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
