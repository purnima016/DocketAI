# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Security: create non-root user
RUN useradd -m -u 1000 docketai

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create empty __init__ files for packages
RUN touch tasks/__init__.py graders/__init__.py

# Change ownership to non-root user
RUN chown -R docketai:docketai /app

# Switch to non-root user
USER docketai

# Expose port
EXPOSE 8000

# Run the environment server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]