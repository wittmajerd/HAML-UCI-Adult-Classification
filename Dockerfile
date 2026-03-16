FROM python:3.12.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

COPY . .

#CMD ["python", "app.py"]