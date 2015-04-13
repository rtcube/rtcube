#include "parser.h"
#include "tokenizer.h"

#include <iostream>
#include <cassert>

using namespace std;


int main(int argc, char** argv)
{
	auto tokens = CubeSQL::tokenize(R"(
SELECT
    5
WHERE
    yob < 1997
AND
    locality = "Warszawa"
ORDER BY
    5 DESC
LIMIT
    5)");

	CubeSQL::parse(tokens);
	cout << "OK" << endl;
}
