# 12.8+ so the default GPU_ARCH (which now includes sm_100/sm_120 Blackwell)
# compiles. Override GPU_ARCH for an older toolkit.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Makefile ./
COPY src/ src/
COPY tools/ tools/
COPY scripts/ scripts/

RUN python3 tools/gen_table.py table.bin && python3 tools/gen_table.py table16.bin --wide && make gpu

# SUFFIX must be provided; NTFY_TOPIC optional (content-free "found" ping)
CMD ["bash", "scripts/entrypoint.sh"]
