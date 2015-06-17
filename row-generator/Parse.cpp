#include <vector>
#include <string.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>

#include "Generator.h"


using namespace std;

namespace Generator{
// cube definition parsing

bool canGenerateFromCube(cube_info *cube) {
    if (!cube->max_vals || !cube->min_vals || !cube->col_type) {
        std::cerr << "Variables for random generation are not set" << std::endl;
        return false;
    }
    return true;
}

// parses a line of comma separated integer values
void parseIntLine(std::string line, std::vector<int> & vector) {
    std::string elem;
    std::stringstream ss(line);
    int val;

    while (getline(ss, elem, ',')) {
        try {
            val = std::stoi(elem);
            vector.push_back(val);
        }
        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Could not parse '" << elem << "' in line '" << line << std::endl;
        }
    }
}

// returns a pointer to a cube_info struct, parsed from a file
cube_info* LoadCubeFile(std::string filename) {
    std::ifstream file(filename.c_str());
    cube_info* cube = new cube_info();
    if (!file.is_open()) {
        return NULL;
    }

    std::string line;
    int dim_count = 0, m_count = 0;
    // count the number of dimensions and measures
    std::getline(file, line);
    if (line[0] == '#') {
        while (std::getline(file, line) && line[0] != '#') {
            dim_count++;
        }
        while (std::getline(file, line)) {
            m_count++;
        }
    } else {
        // file does not begin with '#'
        return NULL;
    }

    // init the arrays
    int no_cols = dim_count + m_count;
    cube->no_cols = no_cols;
    cube->col_type = new Generator::col_types[no_cols];
    cube->min_vals = new int [no_cols];
    cube->max_vals = new int [no_cols];
    cube->lists = new std::vector<int>[no_cols];

    //now lets read the data
    file.clear();
    file.seekg(0, std::ios::beg);

    int str_index, len, i = 0;
    while (std::getline(file, line)) {
        if (line[0] != '#') {
            if (line[0] == '[') {
                // range of values
                cube->col_type[i] = Generator::range_of_vals;
                int val;
                std::string val_substr;

                str_index = line.find_first_of(',');
                cube->min_vals[i] = std::stoi(line.substr(1, str_index - 1));
                len = line.find_first_of(']') - str_index - 1;
                val_substr = line.substr(str_index + 1, len);

                try {
                    val = std::stoi(val_substr);
                    cube->max_vals[i] = val - cube->min_vals[i];
                }
                catch(const std::invalid_argument& ia)
                {
                    std::cerr << "Could not parse '" << val_substr << "' in line '" << line << std::endl;
                }
            }
            else if (line[0] == 'f'){
                // function
                cube->col_type[i] = Generator::function_vals;
            }
            else {
                // list of values
                cube->col_type[i] = Generator::list_of_vals;

                std::vector<int> list;
                parseIntLine(line, list);
                cube->lists[i] = list;
            }
            ++i;
        }
    }

    return cube;
}

}
