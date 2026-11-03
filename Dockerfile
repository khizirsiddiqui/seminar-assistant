FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/khizirsiddiqui/seminar-assistant" \
      org.opencontainers.image.description="Seminar Assistant - schedule to ICS calendar and AI reading guides"

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY templates/ templates/

RUN useradd --create-home appuser && \
    mkdir -p data/sessions && \
    chown -R appuser:appuser /srv

USER appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]