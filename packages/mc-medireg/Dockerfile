# 构建上下文必须是仓库根目录：
#   docker build -f packages/mc-medireg/Dockerfile -t mc-medireg .
# 因为要同时装入兄弟包 mc-domain（单仓库共享底座）。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装底座，再装服务本体。--no-deps 是为了让 pip 不必去索引里找
# 本地包名 "mc-domain"，依赖在这里显式列出，构建结果可复现。
COPY packages/mc-domain /tmp/build/mc-domain
COPY packages/mc-medireg /tmp/build/mc-medireg
RUN pip install --no-cache-dir /tmp/build/mc-domain \
 && pip install --no-cache-dir --no-deps /tmp/build/mc-medireg \
 && pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" \
 && rm -rf /tmp/build

RUN useradd --uid 10001 --create-home appuser
USER appuser

EXPOSE 8011

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8011/healthz', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "mc_medireg.main:app", "--host", "0.0.0.0", "--port", "8011"]
