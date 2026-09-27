# =========================================================================
# Production Dockerfile: Explainable Credit Underwriting & Default Scoring
# Multi-stage optimized, non-root user, Linux libgomp1 for XGBoost
# =========================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies required by XGBoost (libgomp1) and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install locked Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root application user for container security
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy application source code, pre-trained model artifacts, static UI, and schema
COPY src/ ./src/
COPY static/ ./static/
COPY model/ ./model/
COPY database.sql .
COPY main.py .

# Ensure appropriate ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose internal listening port
EXPOSE 8000

# Container healthcheck querying API readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
