FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Makefile ./
COPY src/ src/
COPY tools/ tools/
COPY scripts/ scripts/

RUN python3 tools/gen_table.py table.bin && make gpu

# SUFFIX must be provided; NTFY_TOPIC optional (content-free "found" ping)
CMD ["bash", "scripts/entrypoint.sh"]
