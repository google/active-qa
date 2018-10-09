#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

WORK_DIR=./px/
echo "WORK_DIR: " $WORK_DIR

pushd $WORK_DIR || exit 1
  python -m grpc_tools.protoc -I./proto \
  --python_out=./proto --grpc_python_out=./proto ./proto/aqa.proto
  python -m grpc_tools.protoc -I./proto \
  --python_out=./proto --grpc_python_out=./proto ./proto/reformulator.proto
popd
