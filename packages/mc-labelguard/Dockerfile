FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY packages/mc-domain /tmp/build/mc-domain
COPY packages/mc-labelguard /tmp/build/mc-labelguard
RUN pip install --no-cache-dir /tmp/build/mc-domain \
 && pip install --no-cache-dir --no-deps /tmp/build/mc-labelguard \
 && pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" \
 && rm -rf /tmp/build

RUN useradd --uid 10001 --create-home appuser
USER appuser

EXPOSE 8012

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8012/healthz', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "mc_labelguard:app", "--host", "0.0.0.0", "--port", "8012"]
