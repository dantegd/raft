# syntax=docker/dockerfile:1

ARG CUDA_VER=12.0.1
ARG PYTHON_VER=3.10
ARG LINUX_VER=ubuntu22.04
ARG RAPIDS_VER=23.10

FROM rapidsai/ci:latest
ARG CUDA_VER
ARG PYTHON_VER

ARG AWS_ACCESS_KEY_ID=""
ARG AWS_SECRET_ACCESS_KEY=""
ARG AWS_SESSION_TOKEN=""
ARG RAPIDS_ARTIFACTS_DIR=""

ARG RAPIDS_VER

ARG RAPIDS_BRANCH="branch-${RAPIDS_VER}"

RUN git clone https://github.com/rapidsai/raft.git -b ${RAPIDS_BRANCH}

WORKDIR /raft

RUN printenv

RUN ci/build_cpp.sh

RUN ci/build_python.sh
