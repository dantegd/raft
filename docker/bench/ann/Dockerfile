# syntax=docker/dockerfile:1

ARG CUDA_VER=12.0.1
ARG PYTHON_VER=3.10
ARG LINUX_VER=ubuntu22.04
ARG RAPIDS_VER=23.10
# ARG RAFT_BRANCH="branch-${RAPIDS_VER}"
ARG RAFT_BRANCH="dev-dockerbench"
ARG RAFT_FORK="dantegd"

FROM rapidsai/ci:cuda12.0.1-ubuntu22.04-py3.10
ARG CUDA_VER
ARG PYTHON_VER
ARG LINUX_VER
ARG RAPIDS_VER
ARG RAFT_BRANCH
ARG RAFT_FORK

ARG CI=false
ARG AWS_ACCESS_KEY_ID=false
ARG AWS_SECRET_ACCESS_KEY=false
ARG AWS_SESSION_TOKEN=false
ARG RAPIDS_ARTIFACTS_DIR=""
ARG CMAKE_CUDA_COMPILER_LAUNCHER=sccache=""
ARG CMAKE_CXX_COMPILER_LAUNCHER=sccache=""
ARG CMAKE_C_COMPILER_LAUNCHER=sccache=""

RUN git clone https://github.com/${RAFT_FORK}/raft.git -b ${RAFT_BRANCH}

WORKDIR /raft

# RUN printenv

RUN git apply docker/bench/ann/patches/conda_aws_creds.patch \
    ci/build_cpp.sh
# RUN rapids-mamba-retry mambabuild conda/recipes/libraft

# RUN ci/build_python.sh
