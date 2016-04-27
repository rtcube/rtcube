#!/bin/bash
ADDRESS=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
LD_LIBRARY_PATH=lib bin/gpunode cuda experiment/cubedef $ADDRESS:50000 $ADDRESS:50001
