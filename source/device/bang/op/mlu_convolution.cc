#include "bang_executor.hpp"

extern "C" {
#include "convolution_param.h"

#include "graph/tensor.h"
#include "operator/op.h"
#include "utility/log.h"
}

void conv_mlu_kernel(cnnlHandle_t handle, cnrtQueue_t queue, struct graph* ir_graph, struct node* ir_node, dict_uint2voidx mlu_addr_map,
                     cnnlConvolutionForwardAlgo_t& algo1, int setalgo)
{
    struct tensor* conv_input_data = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* conv_weight = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* conv_output_data = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;

    /* transpose nchw to nhwc */
    if (conv_input_data->layout == 0)
    {
        // transpose input descriptor
        cnnlTensorDescriptor_t tran_input_desc;
        cnnlCreateTensorDescriptor(&tran_input_desc);
        cnnlSetTensorDescriptor(tran_input_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, conv_input_data->dims);

        int permute[4] = {0, 2, 3, 1};
        int tmp = conv_input_data->dims[3];
        conv_input_data->dims[3] = conv_input_data->dims[1];
        conv_input_data->dims[1] = conv_input_data->dims[2];
        conv_input_data->dims[2] = tmp;

        // transpose output descriptor
        cnnlTensorDescriptor_t tran_output_desc;
        cnnlCreateTensorDescriptor(&tran_output_desc);
        cnnlSetTensorDescriptor(tran_output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, conv_input_data->dims);

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
        cnnlTranspose_v2(handle, tran_desc, tran_input_desc, mlu_addr_map[conv_input_data->index], tran_output_desc, mlu_addr_map[conv_input_data->index], 
                         tran_workspace, tran_workspace_size);

        /* destroy transpose descriptor */
        cnnlDestroyTensorDescriptor(tran_input_desc);
        cnnlDestroyTensorDescriptor(tran_output_desc);
        cnnlDestroyTransposeDescriptor(tran_desc);

        /* cnrtFree */
        cnrtFree(tran_workspace);

        conv_input_data->layout = 1;
    }
    if (conv_weight->layout == 0)
    {
        // transpose input descriptor
        cnnlTensorDescriptor_t tran_input_desc;
        cnnlCreateTensorDescriptor(&tran_input_desc);
        cnnlSetTensorDescriptor(tran_input_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, conv_weight->dims);

        int permute[4] = {0, 2, 3, 1};
        int tmp = conv_weight->dims[3];
        conv_weight->dims[3] = conv_weight->dims[1];
        conv_weight->dims[1] = conv_weight->dims[2];
        conv_weight->dims[2] = tmp;

        // transpose output descriptor
        cnnlTensorDescriptor_t tran_output_desc;
        cnnlCreateTensorDescriptor(&tran_output_desc);
        cnnlSetTensorDescriptor(tran_output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, conv_weight->dims);

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
        cnnlTranspose_v2(handle, tran_desc, tran_input_desc, mlu_addr_map[conv_weight->index], tran_output_desc, mlu_addr_map[conv_weight->index], 
                         tran_workspace, tran_workspace_size);

        /* destroy transpose descriptor */
        cnnlDestroyTensorDescriptor(tran_input_desc);
        cnnlDestroyTensorDescriptor(tran_output_desc);
        cnnlDestroyTransposeDescriptor(tran_desc);

        /* cnrtFree */
        cnrtFree(tran_workspace);

        conv_weight->layout = 1;
    }

    // input descriptor
    cnnlTensorDescriptor_t input_desc;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, conv_input_data->dims);

    // filter descriptor
    cnnlTensorDescriptor_t filter_desc;
    cnnlCreateTensorDescriptor(&filter_desc);
    cnnlSetTensorDescriptor(filter_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, conv_weight->dims);

    /* set output data layout and shape */
    if (conv_output_data->layout == 0)
    {
        conv_output_data->layout = 1;
        int tmp = conv_output_data->dims[3];
        conv_output_data->dims[3] = conv_output_data->dims[1];
        conv_output_data->dims[1] = conv_output_data->dims[2];
        conv_output_data->dims[2] = tmp;
    }
    
    // output descriptor
    cnnlTensorDescriptor_t output_desc;
    cnnlCreateTensorDescriptor(&output_desc);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, conv_output_data->dims);

    // convolution descriptor
    int pad[4] = {conv_param->pad_h0, conv_param->pad_h1, conv_param->pad_w0, conv_param->pad_w1};
    int stride[2] = {conv_param->stride_h, conv_param->stride_w};
    int dilation[2] = {conv_param->dilation_h, conv_param->dilation_w};
    int group = conv_param->group;
    cnnlConvolutionDescriptor_t conv_desc;
    cnnlCreateConvolutionDescriptor(&conv_desc);
    cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, group, CNNL_DTYPE_FLOAT);

    /* get algorithm */
    if (0 == setalgo)
    {
        cnnlGetConvolutionForwardAlgorithm(handle, conv_desc, input_desc, filter_desc, output_desc, CNNL_CONVOLUTION_FWD_FASTEST, &algo1);
    }

    /* malloc workspace */
    size_t workspace_size = 0;
    cnnlGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, output_desc, nullptr, conv_desc, algo1, &workspace_size);
    void* workspace = nullptr;
    cnrtMalloc(&workspace, workspace_size);

    /* convolution forward run */
    if (2 < ir_node->input_num)
    {
        struct tensor* conv_bias = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

        cnnlTensorDescriptor_t bias_desc;
        cnnlCreateTensorDescriptor(&bias_desc);
        cnnlSetTensorDescriptor(bias_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, 1, conv_bias->dims);

        cnnlConvolutionForward(handle, conv_desc, algo1, nullptr, input_desc, mlu_addr_map[conv_input_data->index],
                               filter_desc, mlu_addr_map[conv_weight->index], bias_desc, mlu_addr_map[conv_bias->index],
                               workspace, workspace_size, nullptr, output_desc, mlu_addr_map[conv_output_data->index]);

        cnnlDestroyTensorDescriptor(bias_desc);
    }
    else
    {
        cnnlConvolutionForward(handle, conv_desc, algo1, nullptr, input_desc, mlu_addr_map[conv_input_data->index],
                               filter_desc, mlu_addr_map[conv_weight->index], nullptr, nullptr, workspace, workspace_size,
                               nullptr, output_desc, mlu_addr_map[conv_output_data->index]);
    }

    /* destroy descriptors*/
    cnnlDestroyTensorDescriptor(input_desc);
    cnnlDestroyTensorDescriptor(filter_desc);
    cnnlDestroyTensorDescriptor(output_desc);
    cnnlDestroyConvolutionDescriptor(conv_desc);

    /* free workspace */
    cnrtFree(workspace);

    /* relu */
    if (conv_param->activation == 0)
    {
        // relu input descriptor
        cnnlTensorDescriptor_t relu_input_desc;
        cnnlCreateTensorDescriptor(&relu_input_desc);
        cnnlSetTensorDescriptor(relu_input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, conv_output_data->dims);

        // relu descriptor
        cnnlActivationDescriptor_t relu_desc;
        cnnlCreateActivationDescriptor(&relu_desc);
        cnnlSetActivationDescriptor_v5(relu_desc, CNNL_ACTIVATION_RELU, CNNL_ACTIVATION_FAST, CNNL_NOT_PROPAGATE_NAN, 0, 0, 0, 0, false);

        // relu output descriptor
        cnnlTensorDescriptor_t relu_output_desc;
        cnnlCreateTensorDescriptor(&relu_output_desc);
        cnnlSetTensorDescriptor(relu_output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, conv_output_data->dims);

        cnnlActivationForward(handle, relu_desc, nullptr, relu_input_desc, mlu_addr_map[conv_output_data->index], nullptr, relu_output_desc, mlu_addr_map[conv_output_data->index]);
    }
}

void BANGEngine::AddConvolutionNode(struct graph* ir_graph, struct node* ir_node)
{
    TLOG_INFO("Tengine MLU: Support OP(%d) OP_CONV.\n", ir_node->index);
    this->ops.push_back(std::bind(&conv_mlu_kernel, this->cnnl_handle, this->cnrt_queue, ir_graph, ir_node, this->mlu_addr_map, this->algo1, 0));
}
