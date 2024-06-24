# A template of a Dockerfile for a Python project using poetry
# *** TODO:  Edit this Dockerfile to the specific requirements of the application  *** 

FROM python:3.11-bullseye as build

ARG CODEARTIFACT_TOKEN

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH=/usr/bin/poetry/bin:$PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
    POETRY_HOME=/usr/bin/poetry

# Python version must be 3.5 or higher
# Poetry must version be 1.1.7 or higher
RUN curl -sSL https://install.python-poetry.org > ./install-poetry.py && \
    python ./install-poetry.py && \
    rm ./install-poetry.py

    # Install dependencies and the source code of the pipeline
COPY . .
RUN POETRY_HTTP_BASIC_CODEARTIFACT_PASSWORD=${CODEARTIFACT_TOKEN} poetry update daputils && \
    POETRY_HTTP_BASIC_CODEARTIFACT_PASSWORD=${CODEARTIFACT_TOKEN} poetry lock --no-update && \
    POETRY_HTTP_BASIC_CODEARTIFACT_PASSWORD=${CODEARTIFACT_TOKEN} poetry install --only main && \
    poetry run pip install .

# Create virtualenv for deployment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
COPY . .
RUN poetry install --only main

################################################

FROM python:3.11-slim

ARG BUILD_TIMESTAMP
ARG BUILD_VERSION
ARG VERSION

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HOME=/home/pyphot/

LABEL com.ultimagenomics.build-timestamp=${BUILD_TIMESTAMP}
LABEL com.ultimagenomics.build-version=${BUILD_VERSION}
LABEL com.ultimagenomics.version=${VERSION}

# WORKDIR /app

COPY --from=build /opt/venv /opt/venv
# COPY . .

RUN useradd -m pyphot -u 1001 && useradd -m ubuntu -u 1000

USER pyphot
