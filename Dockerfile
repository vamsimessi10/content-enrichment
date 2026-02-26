
# start from official Python image
# slim = smaller size, no unnecessary packages
FROM python:3.11-slim

# set working directory inside container
WORKDIR /app

# copy requirements first
# Docker caches this layer — if requirements don't change,
# it won't reinstall everything on every build. faster.
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy your app code
COPY app.py .

# tell Docker this app listens on port 8000
EXPOSE 8000

# command that runs when container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
