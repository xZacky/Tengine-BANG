#pragma once

extern "C" {
#include "device/device.h"
#include "graph/subgraph.h"

int bang_dev_init(struct device* dev);
int bang_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options);
int bang_dev_run(struct device* dev, struct subgraph* subgraph);
int bang_dev_postrun(struct device* dev, struct subgraph* subgraph);
int bang_dev_release(struct device* dev);
}