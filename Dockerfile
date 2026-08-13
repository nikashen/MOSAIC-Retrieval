FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/data/huggingface

WORKDIR /app

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[serve]"

COPY configs ./configs
COPY reports ./reports
COPY scripts ./scripts
COPY integrations ./integrations
COPY run_project.ps1 ./

EXPOSE 8050
CMD ["python", "-B", "-m", "mosaic.serving", "--root", "/app", "--host", "0.0.0.0", "--port", "8050", "--device", "cpu"]
