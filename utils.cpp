/// Utilities for image processing with openCL
///


#include "utils.hpp"

///
/// \brief This function prints image format informations.
/// \param format: image formats
void printImageFormat(cl_image_format format)
{
#define CASE(order) case order: cout << #order; break;
  switch (format.image_channel_order)
    {
      CASE(CL_R);
      CASE(CL_A);
      CASE(CL_RG);
      CASE(CL_RA);
      CASE(CL_RGB);
      CASE(CL_RGBA);
      CASE(CL_BGRA);
      CASE(CL_ARGB);
      CASE(CL_INTENSITY);
      CASE(CL_LUMINANCE);
      CASE(CL_Rx);
      CASE(CL_RGx);
      CASE(CL_RGBx);
      CASE(CL_DEPTH);
      CASE(CL_DEPTH_STENCIL);
    }
#undef CASE

  cout << " - ";

#define CASE(type) case type: cout << #type; break;
  switch (format.image_channel_data_type)
    {
      CASE(CL_SNORM_INT8);
      CASE(CL_SNORM_INT16);
      CASE(CL_UNORM_INT8);
      CASE(CL_UNORM_INT16);
      CASE(CL_UNORM_SHORT_565);
      CASE(CL_UNORM_SHORT_555);
      CASE(CL_UNORM_INT_101010);
      CASE(CL_SIGNED_INT8);
      CASE(CL_SIGNED_INT16);
      CASE(CL_SIGNED_INT32);
      CASE(CL_UNSIGNED_INT8);
      CASE(CL_UNSIGNED_INT16);
      CASE(CL_UNSIGNED_INT32);
      CASE(CL_HALF_FLOAT);
      CASE(CL_FLOAT);
      CASE(CL_UNORM_INT24);
    }
#undef CASE

  cout << endl;
}

/// \brief Function to display informations on formats that can be handled by the device
/// \param context: openCL context previously created
void displayImageFormats(cl_context context)
{
  cl_uint uiNumSupportedFormats = 0;

  // 2D
  clGetSupportedImageFormats(context,
			     CL_MEM_WRITE_ONLY,
			     CL_MEM_OBJECT_IMAGE2D,
			     0, NULL, &uiNumSupportedFormats);

  cl_image_format* ImageFormats = new cl_image_format[uiNumSupportedFormats];
  clGetSupportedImageFormats(context,
			     CL_MEM_WRITE_ONLY,
			     CL_MEM_OBJECT_IMAGE2D,
			     uiNumSupportedFormats, ImageFormats, NULL);

  for(unsigned int i = 0; i < uiNumSupportedFormats; i++)
    {

      printImageFormat(ImageFormats[i]);

    }

  delete [] ImageFormats;
}


///
///  \brief Round up to the nearest multiple of the group size
///  this is used to get more work items than data
///  \param groupSize:  local_work_size
///  \param globalSize: global_work_size
///  \return first multiple of the groupSize higher than globalSize
size_t RoundUp(int groupSize, int globalSize)
{
  int r = globalSize % groupSize;
  if(r == 0)
    {
      return globalSize;
    }
  else
    {
      return globalSize + groupSize - r;
    }
}

/// \brief convert the kernel file into a string
/// \param filename: string containing file name with kernels
/// \param s: string written by the function containing the result
/// \return: control value
int convertToString(const char *filename, std::string& s)
{
  size_t size;
  char*  str;
  std::ifstream kernelFile(filename, std::ios::in);
  if (!kernelFile.is_open())
    {
      std::cerr << "Failed to open file for reading: " << filename << std::endl;
      return FAILURE;
    }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  s = oss.str();
  return SUCCESS;
}

/// \brief Get error string by number
/// This is used to display the errors more explicitely
/// \param error: openCL error code
/// \return string containing error message
const char *getErrorString(cl_int error)
{
  switch(error){
    // run-time and JIT compiler errors
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default: return "Unknown OpenCL error";
  }
}

/// \brief Read 8bits image from file, return array
/// This function uses FreeImage for simplicity
/// \param filename: name of the file in disk. all image formats
/// \param width of image
/// \param height of image
/// \param channels: number of channels (should be 4, including alpha channel)
/// \return byte array containing the image
unsigned char *readImageFile(const char *filename, size_t &width, size_t &height, int &channels)
{

  FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename, 0);
  FIBITMAP* image = FreeImage_Load(format, filename);

  // Convert to 32-bit image
  FIBITMAP* temp = image;
  image = FreeImage_ConvertTo32Bits(image);
  FreeImage_Unload(temp);

  width = FreeImage_GetWidth(image);
  height = FreeImage_GetHeight(image);
  channels = 4;

  unsigned char *buffer = new unsigned char[width * height * channels];
  memcpy(buffer, FreeImage_GetBits(image), width * height * channels);

  FreeImage_Unload(image);

  return buffer;
}

/// \brief Save the byte array given by buffer into the file given by filename
/// \param filename: name of the image file to write
/// \param buffer: byte array containing the image
/// \param width of image
/// \param height of image
bool saveImageFile(const char *filename, unsigned char *buffer, size_t width, size_t height)
{
  FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(filename);

  FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE*)buffer,
						 width,
						 height,
						 width*4,
						 32,
						 0xFF000000,
						 0x00FF0000,
						 0x0000FF00);

  return FreeImage_Save(format, image, filename);
}
