FROM python:3.11-slim

# HuggingFace Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies as root first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and fix ownership
COPY --chown=user:user . .

# Switch to non-root user
USER user

EXPOSE 7860

CMD ["python", "app.py"]
