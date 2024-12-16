#pragma once

#include "engine/cusparse.hpp"
#include "engine/engine_base.hpp"
#include "engine/engine_bsr.hpp"
#include "engine/engine_coo.hpp"
#include "engine/engine_csr.hpp"
#include "engine/engine_ell.hpp"

namespace cuspmm {

template <typename EngT>
void runEngine(EngT* engine, typename EngT::MataT* a, typename EngT::MatbT* b, float abs_tol, float rel_tol, bool skipSeq);

}