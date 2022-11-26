#include "bang_executor.hpp"

extern "C" {
#include "graph/tensor.h"
#include "utility/log.h"
}

BANGEngine::BANGEngine()
{
}

void BANGEngine::BANGDataMalloc(struct graph* ir_graph, int ir_tensor_idx)
{
    auto iter = this->mlu_addr_map.find(ir_tensor_idx);
    if (this->mlu_addr_map.end() == iter)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        void* mlu_data = nullptr;
        if (cnrtSuccess == cnrtMalloc((void**)&mlu_data, ir_tensor->elem_num * ir_tensor->elem_size))
        {
            TLOG_INFO(" cnrt malloc tensor(%d) name %s size %d addr %p\n",
                      ir_tensor->index, ir_tensor->name, ir_tensor->elem_num * ir_tensor->elem_size, mlu_data);
        }
        if (TENSOR_TYPE_CONST == ir_tensor->tensor_type || TENSOR_TYPE_DEP == ir_tensor->tensor_type)
        {
            TLOG_INFO(" cnrt copy tensor(%d) name %s addr %p\n", ir_tensor->index, ir_tensor->name, mlu_data);
            cnrtMemcpy(mlu_data, ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size, cnrtMemcpyHostToDev);
        }
        this->mlu_addr_map[ir_tensor_idx] = mlu_data;
    }
}

void BANGEngine::DataUpload(struct graph* ir_graph, int ir_tensor_idx)
{
    struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
    cnrtMemcpy(this->mlu_addr_map[ir_tensor_idx], ir_tensor->data, ir_tensor->elem_num * ir_tensor->elem_size, cnrtMemcpyHostToDev);
}

void BANGEngine::DataDownload(struct graph* ir_graph, int ir_tensor_idx)
{
    struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
    cnrtMemcpy(ir_tensor->data, this->mlu_addr_map[ir_tensor_idx], ir_tensor->elem_num * ir_tensor->elem_size, cnrtMemcpyDevToHost);
}

int BANGEngine::Build(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;

        switch (op_type)
        {
        case OP_CONST:
            break;
        case OP_CONV:
            this->AddConvolutionNode(ir_graph, ir_node);
            break;
        case OP_INPUT:
            break;
        case OP_POOL:
            this->AddPoolingNode(ir_graph, ir_node);
            break;
        default:
            TLOG_INFO("Tengine MLU: Cannot support OP(%d).\n", ir_node->index);
            break;
        }
    }
    return 0;
}

int BANGEngine::BANGEnginePreRun(struct subgraph* subgraph)
{
    const auto cnrt_status = cnrtSetDevice(DEFAULT_DEVICE_ID);
    if (cnrt_status != cnrtSuccess)
    {
        fprintf(stderr, "Tengine: Cannot lock to socket %d.\n", DEFAULT_DEVICE_ID);
        return -1;
    }

    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        for (int j = 0; j < ir_node->input_num; j++)
        {
            int ir_tensor_idx = ir_node->input_tensors[j];
            this->BANGDataMalloc(ir_graph, ir_tensor_idx);
        }
        for (int j = 0; j < ir_node->output_num; j++)
        {
            int ir_tensor_idx = ir_node->output_tensors[j];
            this->BANGDataMalloc(ir_graph, ir_tensor_idx);
        }
    }

    
    /* create queue */
    cnrtQueueCreate(&this->cnrt_queue);
    /* create handle */
    cnnlCreate(&this->cnnl_handle);
    cnnlSetQueue(this->cnnl_handle, this->cnrt_queue);

    this->Build(subgraph);

    for (int i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        struct tensor* graph_out_tensor = get_ir_graph_tensor(ir_graph, ir_tensor_idx);
        graph_out_tensor->data = (void*)malloc(graph_out_tensor->elem_num * graph_out_tensor->elem_size);
    }

    return 0;
};

int BANGEngine::BANGEngineRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    /* upload data */
    for (uint8_t i = 0; i < subgraph->input_num; i++)
    {
        int ir_tensor_idx = subgraph->input_tensor_list[i];
        this->DataUpload(ir_graph, ir_tensor_idx);
    }

    /* run */
    for (auto& func : this->ops)
    {
        func();
    }

    /* download data */
    cnrtQueueSync(this->cnrt_queue);
    for (uint8_t i = 0; i < subgraph->output_num; i++)
    {
        int ir_tensor_idx = subgraph->output_tensor_list[i];
        this->DataDownload(ir_graph, ir_tensor_idx);
    }

#ifdef DEBUG_DATA
    for (auto iter = this->mlu_addr_map.begin(); iter != this->mlu_addr_map.end(); iter++)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, iter->first);
        if (ir_tensor->data != NULL)
        {
            cnrtMemcpy(ir_tensor->data, iter->second, ir_tensor->elem_num * ir_tensor->elem_size, cnrtMemcpyDevToHost);
        }
    }
#endif

    return 0;
}

void BANGEngine::BANGEnginePostRun()
{
    for (auto iter = this->mlu_addr_map.begin(); iter != this->mlu_addr_map.end(); iter++)
    {
        cnrtFree(iter->second);
    }
    cnnlDestroy(this->cnnl_handle);
    cnrtQueueDestroy(this->cnrt_queue);
};
