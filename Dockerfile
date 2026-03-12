# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip before installing requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]