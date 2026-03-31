FROM python:3.12.13-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000",     \
     "--backend-store-uri", "file:/app/mlruns", \
     "--default-artifact-root", "file:/app/mlruns"]