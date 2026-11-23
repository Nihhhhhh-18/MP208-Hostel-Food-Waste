# Official lightweight Python 3.12 base image
FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing bytecode and enable unbuffered stdout streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Step 1: Copy and install pinned requirements to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Copy application, packages, model artifacts, and configs
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .
COPY pytest.ini .

# Step 3: Expose container port
EXPOSE 8000

# Step 4: Run uvicorn on container start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
