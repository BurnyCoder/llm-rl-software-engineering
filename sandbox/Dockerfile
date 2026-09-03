# syntax=docker/dockerfile:1
# Global context: run one model-produced Python function behind a minimal boundary.
# Source: https://docs.docker.com/reference/dockerfile/

# Pin the official multi-platform Python image by digest for reproducible rebuilds.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Avoid writes beside source files and make the JSON protocol immediately visible.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep the immutable runner at one predictable path inside the image.
WORKDIR /opt/minibug

# Copy only the protocol runner; candidate source arrives through standard input.
COPY --chown=65532:65532 runner.py /opt/minibug/runner.py

# A numeric unprivileged identity works without adding mutable account metadata.
# Source: https://docs.docker.com/reference/dockerfile/#user
USER 65532:65532

# Isolated mode ignores user-controlled Python environment and site configuration.
# Source: https://docs.python.org/3/using/cmdline.html#cmdoption-I
ENTRYPOINT ["python", "-I", "/opt/minibug/runner.py"]
