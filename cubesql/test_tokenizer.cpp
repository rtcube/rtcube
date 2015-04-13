#include "tokenizer.h"

#include <iostream>
#include <cassert>

using namespace std;


int main(int argc, char** argv)
{
	auto tokens = CubeSQL::tokenize(R"(
SELECT
    locality, yob, SUM(signatures)
FROM
    elections2015
WHERE
    yob IN <1980, 1997>
AND
    locality IN {"Warszawa", "Kraków"}
AND
    pesel[10] IN {1, 3, 5, 7, 9}
ORDER BY
    SUM(signatures) DESC
LIMIT
    5)");

	auto correct_tokens = std::vector<string>{
		"SELECT",
		"locality", ",", "yob", ",", "SUM", "(", "signatures", ")",
		"FROM",
		"elections2015",
		"WHERE",
		"yob", "IN", "<", "1980", ",", "1997", ">",
		"AND",
		"locality", "IN" ,"{", "\"Warszawa\"", ",", "\"Kraków\"", "}",
		"AND",
		"pesel", "[", "10", "]", "IN", "{", "1", ",", "3", ",", "5", ",", "7", ",", "9", "}",
		"ORDER", "BY",
		"SUM", "(", "signatures", ")", "DESC",
		"LIMIT",
		"5"
	};

	assert(tokens.size() == correct_tokens.size());
	for (size_t i = 0; i < tokens.size(); ++i)
	{
		//cout << "AT:" << tokens[i].code << endl;
		//cout << "CT:" << correct_tokens[i] << endl;
		assert(tokens[i].code == correct_tokens[i]);
	}

	cout << "OK" << endl;
}
