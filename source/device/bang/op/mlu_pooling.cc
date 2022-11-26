#include "bang_executor.hpp"

extern "C" {
#include "pooling_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

void pooling_mlu_kernel(cnnlHandle_t handle, cnrtQueue_t queue, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx mlu_addr_map)
{
    struct tensor* pool_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* pool_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct pool_param* pool_param = (struct pool_param*)ir_node->op.param_mem;

    /* transpose nchw to nhwc */
    if (pool_input_data->layout == 0)
    {
        // transpose input descriptor
        cnnlTensorDescriptor_t tran_input_desc;
        cnnlCreateTensorDescriptor(&tran_input_desc);
        cnnlSetTensorDescriptor(tran_input_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, pool_input_data->dims);

        int permute[4] = {0, 2, 3, 1};
        int tmp = pool_input_data->dims[3];
        pool_input_data->dims[3] = pool_input_data->dims[1];
        pool_input_data->dims[1] = pool_input_data->dims[2];
        pool_input_data->dims[2] = tmp;

        // transpose output descriptor
        cnnlTensorDescriptor_t tran_output_desc;
        cnnlCreateTensorDescriptor(&tran_output_desc);
        cnnlSetTensorDescriptor(tran_output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, pool_input_data->dims);

        // transpose descriptor
        cnnlTransposeDescriptor_t tran_desc;
        cnnlCreateTransposeDescriptor(&tran_desc);
        cnnlSetTransposeDescriptor(tran_desc, 4, permute);

        /* malloc transpose workspace */
        size_t tran_workspace_size = 0;
        cnnlGetTransposeWorkspaceSize(handle, tran_input_desc, tran_desc, &tran_workspace_size);
        void *tran_workspace = nullptr;
        cnrtMalloc(&tran_workspace, tran_workspace_size);

        /* run transpose */
        cnnlTranspose_v2(handle, tran_desc, tran_input_desc, mlu_addr_map[pool_input_data->index], tran_output_desc, mlu_addr_map[pool_input_data->index], 
                         tran_workspace, tran_workspace_size);

        /* destroy transpose descriptor */
        cnnlDestroyTensorDescriptor(tran_input_desc);
        cnnlDestroyTensorDescriptor(tran_output_desc);
        cnnlDestroyTransposeDescriptor(tran_desc);

        /* cnrtFree */
        cnrtFree(tran_workspace);

        pool_input_data->layout = 1;
    }

    // input descriptor
    cnnlTensorDescriptor_t input_desc;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, pool_input_data->dims);

    /* set output data layout and shape */
    if (pool_output_data->layout == 0)
    {
        pool_output_data->layout = 1;
        int tmp = pool_output_data->dims[3];
        pool_output_data->dims[3] = pool_output_data->dims[1];
        pool_output_data->dims[1] = pool_output_data->dims[2];
        pool_output_data->dims[2] = tmp;
    }

    // output descriptor
    cnnlTensorDescriptor_t output_desc;
    cnnlCreateTensorDescriptor(&output_desc);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, pool_output_data->dims);

    // pooling descriptor
    cnnlPoolingMode_t poolmode;

    switch (pool_param->pool_method)
    {
    case 0:
        poolmode = CNNL_POOLING_MAX;
        break;
    case 1:
        poolmode = CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
    default:
        fprintf(stderr, "don't support this method pooling\n");
    }
    cnnlPoolingDescriptor_t pool_desc;
    cnnlCreatePoolingDescriptor(&pool_desc);
    cnnlSetPooling2dDescriptor_v2(pool_desc, poolmode, CNNL_NOT_PROPAGATE_NAN, pool_param->kernel_h, pool_param->kernel_w, pool_param->pad_h0,
                                  pool_param->pad_h1, pool_param->pad_w0, pool_param->pad_w1, pool_param->stride_h, pool_param->stride_w, 1, 1, false);

    /* malloc workspace */
    size_t workspace_size = 0;
    cnnlGetPoolingWorkspaceSize(handle, poolmode, pool_output_data->dims[2], pool_output_data->dims[1], &workspace_size);
    void* workspace = nullptr;
    cnrtMalloc(&workspace, workspace_size);

    /* extra input */
    size_t extra_input_size = 0;
    cnnlGetPoolingExtraInputSize(handle, poolmode, pool_output_data->dims[2], pool_output_data->dims[1], &extra_input_size);
    void* extra_host_input = nullptr;
    extra_host_input = (void*)malloc(extra_input_size);
    cnnlInitPoolingExtraInput(handle, pool_desc, input_desc, output_desc, extra_host_input);
    void* extra_dev_input = nullptr;
    cnrtMalloc(&extra_dev_input, extra_input_size);
    cnrtMemcpy(extra_dev_input, extra_host_input, extra_input_size, cnrtMemcpyHostToDev);

    /* pooling forward run */
    cnnlPoolingForward_v2(handle, pool_desc, nullptr, input_desc, mlu_addr_map[pool_input_data->index], nullptr, extra_dev_input, output_desc, mlu_addr_map[pool_output_data->index],
                       workspace, workspace_size);

    /* destroy descriptors*/
    cnnlDestroyTensorDescriptor(input_desc);
    cnnlDestroyTensorDescriptor(output_desc);
    cnnlDestroyPoolingDescriptor(pool_desc);

    /* free workspace */
    cnrtFree(workspace);

    /* free extra input */
    cnrtFree(extra_dev_input);
    free(extra_host_input);
}

void BANGEngine::AddPoolingNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine MLU: Support OP(%d) OP_POOL.\n", ir_node->index);
    this->ops.push_back(std::bind(&pooling_mlu_kernel, this->cnnl_handle, this->cnrt_queue, ir_graph, ir_node, this->mlu_addr_map));
}
