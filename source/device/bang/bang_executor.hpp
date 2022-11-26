#pragma once

#include <map>
#include <vector>
#include <functional>
#include <cstdio>

extern "C" {
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
}

#include "cnrt.h"
#include "cnnl.h"

#define DEFAULT_DEVICE_ID 0

typedef std::map<uint32_t, uint32_t> dict_uint2uint;
typedef std::map<uint32_t, void*> dict_uint2voidx;
typedef std::function<void()> MLU_kernel;

class BANGEngine
{
public:
    BANGEngine();
    ~BANGEngine() = default;

    int BANGEnginePreRun(struct subgraph* subgraph);
    int BANGEngineRun(struct subgraph* subgraph);
    void BANGEnginePostRun();

private:
    void AddConvolutionNode(struct graph* ir_graph, struct node* ir_node);
    void AddPoolingNode(struct graph* ir_graph, struct node* ir_node);

private:
    void BANGDataMalloc(struct graph* ir_graph, int ir_tensor_idx);
    int Build(struct subgraph* subgraph);
    void DataUpload(struct graph* ir_graph, int ir_tensor_idx);
    void DataDownload(struct graph* ir_graph, int ir_tensor_idx);

private:
    std::vector<MLU_kernel> ops;

private:
    cnnlHandle_t cnnl_handle;
    cnrtQueue_t cnrt_queue;
    cnnlConvolutionForwardAlgo_t algo1;

public:
    dict_uint2voidx mlu_addr_map;
};
