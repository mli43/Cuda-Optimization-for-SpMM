#pragma once

#include <chrono>
#include <type_traits>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "utils.hpp"

#define RowMjIdx(x, y, B) ((x) * (B) + (y))
#define ColMjIdx(x, y, A) ((y) * (A) + (x))

#define assertTypes3(T, ta, tb, tc) static_assert(std::is_same_v<T, ta> || std::is_same_v<T, tb> || std::is_same_v<T, tc>, "Unsupported type")