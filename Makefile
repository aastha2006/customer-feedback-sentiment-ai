
.PHONY: install train run test clean docker-build docker-run

install:
	pip install -r requirements.txt

download:
	python -m src.download_data

train:
	python -m src.train

run:
	python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	python -m streamlit run src/ui.py

test:
	python -m pytest tests/

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf models/*.pkl
	rm -rf models/*.joblib
	rm -rf logs/*.log

docker-build:
	docker build -t customer-feedback-ai .

docker-run:
	docker run -p 8000:8000 customer-feedback-ai
