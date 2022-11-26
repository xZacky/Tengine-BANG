#pragma once

extern "C" {
#include "operator/op.h"
}

const int bang_supported_ops[] = {
    OP_CONST,
    OP_CONV,
    OP_INPUT,
    OP_POOL};
