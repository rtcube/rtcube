#!/bin/bash
GENERATOR=bin/data-generator
FILE=addresses_all_udp
NUMBER=4

if [$# -gt 2]; then
	FILE=$2
	
