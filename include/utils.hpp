#pragma once

#include <string>

inline bool endsWith(const std::string &fullString, const std::string &ending) {
    if (ending.size() > fullString.size())
        return false;

    // Compare the ending of the full string with the target
    // ending
    return fullString.compare(fullString.size() - ending.size(), ending.size(),
                              ending) == 0;
}
