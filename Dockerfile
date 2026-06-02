# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy requirements file to container
COPY requirements.txt .

# Install packages to user local directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.10-slim AS runner

WORKDIR /workspace

# Create a non-privileged system group and user for security compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /sbin/nologin appuser

# Copy dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set environment paths and disable python output buffering
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy source code and models
COPY ./app ./app

# Set ownership of workspace to non-root user
RUN mkdir -p app/model && chown -R appuser:appgroup /workspace

# Use the non-root user
USER appuser

# Expose port
EXPOSE 8000

# Start server using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
