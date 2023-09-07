#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

python -m raft-ann-bench.get_dataset $1

python -m raft-ann-bench.run $1

