#!/bin/bash

if [ $# -lt 2 ]
then
  echo "Error, missing arguments"
  echo "USAGE: experiment.sh <addresses_for_querer> <addresses_for_generator>"
  exit 1
fi

python3 auto_querer.py $1 experiment/cudebef && bin/data-generator $2
