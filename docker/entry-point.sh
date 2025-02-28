#!/bin/bash

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

if [ ! -z "$VIVADO_PATH" ]; then
    source $VIVADO_PATH/settings64.sh
else
    echo "Warning: \$VIVADO_PATH not defined. Continuing but without synthesis support."
fi

pip install -e /workspace/logicnets[example-all]

exec "$@"

