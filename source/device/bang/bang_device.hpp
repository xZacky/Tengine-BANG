#pragma once

#include "device/device.h"

#define BANG_DEV_NAME "BANG"

extern "C" {
struct bang_device
{
    struct device base;
};

int register_bang_device(void);
}
