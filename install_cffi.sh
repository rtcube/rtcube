#!/bin/bash
DIR=~/local/lib/python3.5/site-packages
if [ ! -d "$DIR" ]; then
	mkdir -p $DIR
fi
export PYTHONPATH=$DIR
easy_install --prefix=$HOME/local cffi
