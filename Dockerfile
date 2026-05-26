FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs data/raw data/processed

EXPOSE 8501

CMD ["bash", "-c", "python pipeline.py && streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"]
