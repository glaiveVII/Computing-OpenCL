#ifndef UTILS_CPP
#define UTILS_CPP
/*
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif
*/

#include <CL/opencl.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

// This code uses the FreeImage library
// http://freeimage/sourceforge.net
// sudo apt-get install libfreeimage-dev
#include "FreeImage.h"

using namespace std;

#define SUCCESS 0
#define FAILURE 1

using namespace std;

void printImageFormat(cl_image_format format);
void displayImageFormats(cl_context context);
size_t RoundUp(int groupSize, int globalSize);
int convertToString(const char *filename, std::string& s);
const char *getErrorString(cl_int error);
unsigned char *readImageFile(const char *filename, size_t &width, size_t &height, int &channels);
bool saveImageFile(const char *filename, unsigned char *buffer, size_t width, size_t height);


#endif
