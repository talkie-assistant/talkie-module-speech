# Standalone build: clones talkie-core for sdk + modules.api, overlays this repo as modules/speech.
# Image matches CLI/compose: ghcr.io/talkie-assistant/talkie-module-speech, CMD python -m modules.speech.server
FROM alpine/git as talkie-core
ARG TALKIE_CORE_REF=main
WORKDIR /src
RUN git clone --depth 1 --branch "${TALKIE_CORE_REF}" https://github.com/talkie-assistant/talkie-core.git .

FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=talkie-core /src/sdk ./sdk
COPY --from=talkie-core /src/modules/__init__.py /src/modules/discovery.py ./modules/
COPY --from=talkie-core /src/modules/api ./modules/api
COPY . ./modules/speech
RUN pip install --no-cache-dir -r modules/speech/requirements.txt
RUN mkdir -p data

CMD ["python", "-m", "modules.speech.server", "--host", "0.0.0.0", "--port", "8001"]
