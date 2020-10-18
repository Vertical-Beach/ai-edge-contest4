#!bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)

set -e -o pipefail

mkdir -p $SCRIPT_DIR/build && cd $SCRIPT_DIR/build
cmake .. && make
./gen_dat flip

echo "-- Generated dat files"
