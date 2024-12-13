#include "cusparse.h"

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
                    __LINE__, cusparseGetErrorString(status), status);         \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

