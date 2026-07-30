#!/bin/bash
DIR="$(dirname "$0")"
cd "$DIR"
[ -f trunk.dae ] && exit 0
echo "Reassembling..."
cat trunk.dae.part-* > trunk.dae
rm -f trunk.dae.part-*
echo "Done"
