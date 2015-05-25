#include "tokenizer.h"
#include <stdexcept>

namespace CubeSQL {

static bool is_special_char(char x)
{
	switch (x)
	{
		case '(':
		case ')':
		case '<':
		case '>':
		case '{':
		case '}':
		case '[':
		case ']':
		case ',':
			return true;
	}
	return false;
}

static bool is_space(char x)
{
	switch (x)
	{
		case ' ':
		case '\n':
		case '\t':
			return true;
	}
	return false;
}

static bool is_quote(char x)
{
	switch (x)
	{
		case '"':
		case '\'':
			return true;
	}
	return false;
}

// TODO support \' (or sth like that)
void read_string(token& t, const std::string& data, size_t& i)
{
	auto begin = i;
	auto quote = data[begin];
	++i;
	while (i < data.size() && data[i] != quote)
		++i;
	if (data[i] != quote) // EOF
		throw std::invalid_argument("CubeSQL::tokenize");
	++i;

	t.code = data.substr(begin, i-begin);
	t.val = t.code.substr(1, t.code.size()-2);
}

void try_parse_as_number(token& t)
{
	bool dot = false;
	for (auto c : t.code)
	{
		if (c == '.')
		{
			if (!dot)
				dot = true;
			else
				return; // Two dots!
		}
		else if (!isdigit(c))
			return;
	}

	if (!dot)
		t.val = std::stoll(t.code);
	else
		t.val = std::stod(t.code);
}

auto tokenize(const std::string& data) -> std::vector<token>
{
	auto i = size_t{0};
	auto tokens = std::vector<token>{};

	while (i < data.size())
	{
		token t;

		if (is_space(data[i]))
		{
			++i;
			continue;
		}

		if (is_special_char(data[i]))
		{
			t.code = {data[i++]};
			tokens.push_back(t);
			continue;
		}

		if (is_quote(data[i]))
		{
			read_string(t, data, i);
			tokens.push_back(t);
			continue;
		}

		// Read until special char / quote / space / EOF.
		auto begin = i;
		++i;
		while (i < data.size() && !is_special_char(data[i]) && !is_quote(data[i]) && !is_space(data[i]))
			++i;
		t.code = data.substr(begin, i-begin);
		try_parse_as_number(t);
		tokens.push_back(t);
		continue;
	}

	return tokens;
}

}
