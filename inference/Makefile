install:
	python -m pip install -r requirements.txt

install_dev:
	python -m pip install -r requirements-dev.txt

test: install_dev
	python -m pytest tests -vvv --disable-warnings --cov=housing_inference --cov-report=term-missing

run:
	uvicorn housing_inference.app:app --host 0.0.0.0 --port 8080
