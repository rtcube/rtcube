#!/bin/bash
if [ $# -lt 4 ]
then
  echo "Error, missing arguments"
  echo "USAGE: experiment.sh <addresses_for_querer> <addresses_for_generator> <querer_output> <generator_output>"
  exit 1
fi

python3 auto_querer.py $1 experiment/cudebef > $3 &
QPID=$!
bin/data-generator $2 > $4 &
GPID=$!

trap 'kill $QPID ; kill $GPID' SIGINT
wait
