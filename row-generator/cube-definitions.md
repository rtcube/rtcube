The files defining cube measures and dimensions should have the following structure:

- first row begginning with a hash ('#'), indicating start of dimensions
- followed by rows defining dimensions
- another hash-starting row begins list of measure defining rows

- structure of a row:
for value ranges: [min_value, max_value]
for list of values (comma separated): val1,val2,....,valn

string values have to be enclosed in double quotation marks (") 
	1. only list ranges available for strings
	2. not implemented yet

sample cube definition (see 'cube' file)
