# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#!/bin/bash
set -e


QUAKE_DIR="../Dialect/Quake"
CC_DIR="../Dialect/CC"


echo "Transforming files in $QUAKE_DIR"

# Detect platform and set sed flags accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS requires an empty string for -i backup
  SED_CMD=(sed -i '')
else
  # Linux allows -i without backup suffix
  SED_CMD=(sed -i)
fi

# 1
TARGET_STRING='"cudaq/Optimizer/'
REPLACEMENT_STRING='"'

find "$QUAKE_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" \
  "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +


find "$CC_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" \
  "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +



# Specific changes
TARGET_DIR="../Dialect/Common"
TARGET_STRING='#include "cudaq/Frontend/nvqpp/AttributeNames.h"'
REPLACEMENT_STRING='#include "Dialect/Common/AttributeNames.h"'
find "$TARGET_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +


# Specific changes
TARGET_STRING='#include "cudaq/Support/SmallVector.h"'
REPLACEMENT_STRING='#include "Dialect/Quake/SmallVector.h"'
find "$QUAKE_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +

# Specific changes
TARGET_STRING='#include "cudaq/Support/SmallVector.h"'
REPLACEMENT_STRING='#include "Dialect/Quake/SmallVector.h"'
find "$CC_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +


# Specific changes
TARGET_STRING='#include "Builder/Factory.h"'
REPLACEMENT_STRING='#include "Dialect/Quake/Factory.h"'
find "$QUAKE_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +

# Specific changes
TARGET_STRING='#include "Builder/Intrinsics.h"'
REPLACEMENT_STRING='#include "Dialect/Quake/Intrinsics.h"'
find "$QUAKE_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +

TARGET_STRING='#include "Builder/Runtime.h"'
REPLACEMENT_STRING='#include "Dialect/Quake/Runtime.h"'
find "$QUAKE_DIR" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -exec "${SED_CMD[@]}" "s|$TARGET_STRING|$REPLACEMENT_STRING|g" {} +


echo "All matching includes transformed successfully."
