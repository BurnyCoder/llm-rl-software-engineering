# syntax=docker/dockerfile:1
# Global context: run one function-call or assertion-script request behind Docker.
# Source: https://docs.docker.com/reference/dockerfile/

# Pin the official multi-platform Python image by digest to fix the base filesystem bytes.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Avoid bytecode writes, flush the JSON protocol, and keep NumPy within the PID limit.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# HumanEvalPack's pinned Python harness prepends NumPy imports to every candidate.
# Sources: https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py#L371-L375 and https://pypi.org/project/numpy/2.5.2/
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    --root-user-action=ignore "numpy==2.5.2"

# Keep the immutable runner at one predictable path inside the image.
WORKDIR /opt/minibug

# Copy only the two-protocol runner; all candidate and test source arrives through stdin.
COPY --chown=65532:65532 runner.py /opt/minibug/runner.py

# A numeric unprivileged identity works without adding mutable account metadata.
# Source: https://docs.docker.com/reference/dockerfile/#user
USER 65532:65532

# Isolated mode ignores user-controlled Python environment and site configuration.
# Source: https://docs.python.org/3.12/using/cmdline.html#cmdoption-I
ENTRYPOINT ["python", "-I", "/opt/minibug/runner.py"]
