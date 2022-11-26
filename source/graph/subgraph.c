/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#include "graph/subgraph.h"

#include "utility/sys_port.h"
#include "device/device.h"
#include "api/c_api.h"
#include "utility/log.h"

void init_ir_subgraph(struct graph* graph, struct subgraph* subgraph, int index)
{
    subgraph->index = index;
    subgraph->input_ready_count = 0;
    subgraph->input_wait_count = 0;
    subgraph->input_num = 0;
    subgraph->output_num = 0;
    subgraph->node_num = 0;
    subgraph->node_list = NULL;
    subgraph->input_tensor_list = NULL;
    subgraph->output_tensor_list = NULL;
    subgraph->graph = graph;
    subgraph->device = NULL;
    subgraph->device_graph = NULL;
    subgraph->status = GRAPH_STAT_CREATED;
}

void release_ir_subgraph(struct graph* graph, struct subgraph* subgraph)
{
    struct device* device = subgraph->device;

    if (NULL != subgraph->device_graph && NULL != device->interface && NULL != device->interface->release_graph)
    {
        device->interface->release_graph(device, (void*)subgraph->device_graph);
    }

    sys_free(subgraph->input_tensor_list);
    sys_free(subgraph->output_tensor_list);
    sys_free(subgraph->node_list);
    sys_free(subgraph);
}

void dump_ir_subgraph(struct subgraph* subgraph)
{
    TLOG_INFO("\nsubgraph index: %d\n", subgraph->index);
    struct device* device = subgraph->device;
    TLOG_INFO("\nsubgraph device name: %s\n", device->name);
    TLOG_INFO("\nsubgraph nodes: %d\n", subgraph->node_num);
    for (int i = 0; i < subgraph->node_num; i++)
    {
        TLOG_INFO("%d  ", subgraph->node_list[i]);
    }
    TLOG_INFO("\nsubgraph input tensors: %d\n", subgraph->input_num);
    for (int i = 0; i < subgraph->input_num; i++)
    {
        TLOG_INFO("%d  ", subgraph->input_tensor_list[i]);
    }
    TLOG_INFO("\nsubgraph output tensors: %d\n", subgraph->output_num);
    for (int i = 0; i < subgraph->output_num; i++)
    {
        TLOG_INFO("%d  ", subgraph->output_tensor_list[i]);
    }
    TLOG_INFO("\n");
}
