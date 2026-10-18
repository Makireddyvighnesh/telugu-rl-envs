FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev --no-editable
RUN useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
# The snapshot is built into the image at a pinned Aksharantar revision, so the server
# starts offline and every deployment serves the same task IDs.
ARG AKSHARANTAR_REVISION=
RUN python -m telugu_transliteration.data.prepare --output /app/snapshot ${AKSHARANTAR_REVISION:+--revision $AKSHARANTAR_REVISION}
ENV TRANSLIT_SNAPSHOT=/app/snapshot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "telugu_transliteration.server.app:create_server", "--factory", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets", "--ws-ping-interval", "120", "--ws-ping-timeout", "600"]
