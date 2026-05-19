FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Create a non-root user (Hugging Face requires user 1000 for security)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies (build-essential needed for some C-extensions like rank-bm25)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files and set ownership to our non-root user
COPY --chown=user:user . /app

# Switch to the non-root user
USER user

# Install the package and dependencies
RUN pip install --no-cache-dir --user .

# Ensure the local bin is in PATH so uvicorn can be found
ENV PATH="/home/user/.local/bin:${PATH}"

# Expose Hugging Face's default port
EXPOSE 7860

# Run the FastAPI app using the factory pattern
CMD ["uvicorn", "finrag.api.app:app", "--host", "0.0.0.0", "--port", "7860", "--factory"]
