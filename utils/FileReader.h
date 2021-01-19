#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
class FileReader
{

public:
    string filename;

    FileReader();
    FileReader(string);
    // vector<vector<string> > getData();
    std::vector<uint64_t> getData();
    std::vector<uint64_t> getCSVData();
    
};

FileReader::FileReader()
{

}

FileReader::FileReader(string filename)
{
    this->filename = filename;
}

std::vector<uint64_t> FileReader::getData() 
{
    std::cout<< "filename: " << this->filename << std::endl;
    std::ifstream file(this->filename, std::ios::binary);

    if (!file.is_open()) 
    {
      std::cerr << "unable to open " << filename << std::endl;
    }

    std::vector<uint64_t> data;
    // Read size.
    uint64_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    data.resize(size);
    // Read values.
    file.read(reinterpret_cast<char*>(data.data()), size*sizeof(uint64_t));
    file.close();
    
  return data;
}