ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MEDLIT_DATABASE_PATH=/app/data/medlit.db

COPY pyproject.toml README.md THIRD_PARTY_NOTICES.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

RUN mkdir -p /app/data

EXPOSE 8765

CMD ["python", "-m", "uvicorn", "medlit_agent.main:app", "--host", "0.0.0.0", "--port", "8765"]