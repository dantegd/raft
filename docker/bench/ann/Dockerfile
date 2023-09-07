# syntax=docker/dockerfile:1

# use 11.8 while CONDA 12.0 ARM packages are ready
ARG CUDA_VER=11.8.0
ARG PYTHON_VER=3.10
ARG LINUX_VER=ubuntu22.04
ARG RAPIDS_VER=23.10
# ARG RAFT_BRANCH="branch-${RAPIDS_VER}"
ARG RAFT_BRANCH="dev-dockerbench-tagged"
ARG RAFT_FORK="dantegd"

FROM rapidsai/ci:cuda${CUDA_VER}-${LINUX_VER}-py${PYTHON_VER}
ARG CUDA_VER
ARG PYTHON_VER
ARG LINUX_VER
ARG RAPIDS_VER
ARG RAFT_BRANCH
ARG RAFT_FORK

RUN rapids-mamba-retry install raft-ann-bench=${RAPIDS_VER} pylibraft=${RAPIDS_VER} -y

ARG CACHEBUST=1
RUN git clone https://github.com/${RAFT_FORK}/raft.git -b ${RAFT_BRANCH}
WORKDIR /raft

# workflow for building from source
# ARG CI=false
# ARG RAPIDS_ARTIFACTS_DIR=""

# RUN git apply docker/bench/ann/patches/conda_aws_creds.patch

# RUN ci/build_cpp.sh

# RUN ci/build_python.sh

# workflow for building using conda packages

# should we provide some datasets pre-baked in the container, or a container with datasets already?

RUN ["chmod", "+x", "docker/bench/ann/scripts/run.sh"]

CMD ["--dataset fashion-mnist-784-euclidean", "--algorithms hnswlib"]

ENTRYPOINT ["docker/bench/ann/scripts/run.sh"]

