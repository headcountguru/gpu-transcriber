# Image for the rented Vast.ai GPU machine. Build context is the repo root:
#   docker build -f docker/Dockerfile .
# Holds no secrets and no gated models: pyannote weights are downloaded at job start with the
# user's own Hugging Face token.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/remote \
    HF_HOME=/opt/hf

# openssh-server: Vast.ai runs sshd for the app's SFTP/SSH session. ffmpeg: audio extraction.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torchvision ships in the base image but nothing here needs it, and a mismatch with torch breaks
# imports. Remove it before the requirements are resolved. The constraints keep pip from replacing
# the base image's torch and torchaudio (see docker/constraints.txt).
COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip uninstall -y torchvision \
    && pip install -c /tmp/constraints.txt -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt

# pyannote.audio 3.4.0 still passes use_auth_token, which huggingface_hub no longer accepts.
# The script exits non-zero (failing the build) if it patches no files.
COPY docker/patch_pyannote.py /tmp/patch_pyannote.py
RUN python /tmp/patch_pyannote.py && rm /tmp/patch_pyannote.py

# Warm the faster-whisper cache so jobs never download it. Do not bake pyannote models (gated).
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"

COPY common/ /app/common/
COPY remote/ /app/remote/

WORKDIR /app
