#ifndef SynData_H
#define SynData_H

#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
class SynData
{

public:
  string filename;

  SynData();
  SynData(string);
  // vector<vector<string> > getData();
  std::vector<uint64_t> getData(string filename);
  void writeData(std::vector<uint64_t>, string filename);
};

SynData::SynData()
{
}

SynData::SynData(string filename)
{
  this->filename = filename;
}

std::vector<uint64_t> SynData::getData(string filename) 
{
  std::ifstream file(filename);
  std::vector<uint64_t> data;
  string line = "";
  while (getline(file, line))
  {
    std::vector<string> vec;
    boost::algorithm::split(vec, line, boost::is_any_of(","));
    data.push_back((uint64_t)stoll(vec[0]));
  }
  file.close();
  return data;
}

void SynData::writeData(std::vector<uint64_t> data, string filename)
{
  std::ofstream out(filename, std::ios_base::trunc | std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "unable to open " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  // Write size.
  const uint64_t size = data.size();
  out.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));
  // Write values.
  out.write(reinterpret_cast<const char*>(data.data()), size*sizeof(uint64_t));
  out.close();
}

#endif