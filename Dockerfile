FROM python:3.13-slim

# Set environment variables
ENV \
    # Freeze Poetry version for reproducible builds
    POETRY_VERSION=2.1.0 \
    # Disable stdout buffering for real-time CI logs
    PYTHONUNBUFFERED=1

# Install Poetry - explicit version prevents breaking changes
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Configure Poetry
RUN \
    # Disable venv (use system Python)
    poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install dependencies using Poetry
RUN poetry install \
    # Disable prompts for CI/CD automation
    --no-interaction \
    # Remove color formatting for clean logs
    --no-ansi \
    # Skip project install for layer caching
    --no-root

# Copy action code
COPY src/ ./src/

# Set the entry point for GitHub Actions
ENTRYPOINT ["python", "/app/src/main.py"]