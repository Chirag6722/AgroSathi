# Dockerfile — Python 3.11
FROM python:3.11-slim

# Install minimal build deps (for numpy/pandas/torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl ca-certificates libatlas-base-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Upgrade pip & wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5000
ENV PORT=5000

# Start the app with gunicorn (production ready)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "3"]
