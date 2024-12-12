#pragma once

#include <chrono>

#define RowMjIdx(x, y, B) ((x) * (B) + (y))
#define ColMjIdx(x, y, A) ((y) * (A) + (x))