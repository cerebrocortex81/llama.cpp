#!/bin/sh
set -o pipefail

CC=$1

build_number="0"
build_commit="unknown"
build_compiler="unknown"
build_target="unknown"

# git is broken on WSL so we need to strip extra newlines
if out=$(git rev-list --count HEAD | tr -d '\n'); then
  build_number=$out
fi

if out=$(git rev-parse --short HEAD | tr -d '\n'); then
  build_commit=$out
fi

if out=$($CC --version | head -1); then
  build_compiler=$out
fi

if out=$($CC -dumpmachine); then
  build_target=$out
fi

echo "#ifndef BUILD_INFO_H"
echo "#define BUILD_INFO_H"
echo
echo "#define BUILD_NUMBER $build_number"
echo "#define BUILD_COMMIT \"$build_commit\""
echo "#define BUILD_COMPILER \"$build_compiler\""
echo "#define BUILD_TARGET \"$build_target\""
echo
echo "#endif // BUILD_INFO_H"
