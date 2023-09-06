#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

python -m raft-ann-bench.get_dataset --dataset fashion-mnist-784-euclidean

python -m raft-ann-bench.run --dataset deep-image-96-inner

