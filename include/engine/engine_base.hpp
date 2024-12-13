#pragma once

namespace cuspmm {

class EngineBase {
public:
    int numKernels;

    virtual void* runKernel(int num, void* _ma, void* _mb, void* _mc) = 0;
};

}