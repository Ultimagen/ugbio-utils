FROM python:3.11-bullseye AS build

ARG OUT_DIR=/tmp/ugbio
ARG MODULE_DIR=omics
RUN mkdir -p $OUT_DIR

COPY ./src/ ./src/

# build module
RUN python -m pip install build
RUN python -m build --outdir $OUT_DIR --wheel ./src/$MODULE_DIR

##################################################################
FROM python:3.11-slim

#ARG USER_ID=2000
#ARG GROUP_ID=$USER_ID
# Create a non-root user and group
#RUN groupadd -g ${GROUP_ID} ugbio && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ugbio
RUN groupadd ugbio && useradd -m ugbio -g ugbio
RUN chown -R ugbio:ugbio /opt /usr/local/bin

USER ugbio

WORKDIR /home/ugbio

ARG OUT_DIR=/tmp/ugbio

# Copy whl and install module
COPY --from=build $OUT_DIR $OUT_DIR

ENV PATH="/home/ugbio/.local/bin:${PATH}"

RUN WHEEL_FILE=$(find ${OUT_DIR} -name "*.whl") && \
    pip install --user "${WHEEL_FILE}[db]"

# for runnig CI (can be removed if pytest is installed as part of the venv)
RUN pip install --user pytest
