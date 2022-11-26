#include "bang_graph.hpp"
#include "bang_executor.hpp"


int bang_dev_init(struct device* dev)
{
    (void)dev;
    return 0;
}


int bang_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    subgraph->device_graph = new BANGEngine;
    auto engine = (BANGEngine*)subgraph->device_graph;

    return engine->BANGEnginePreRun(subgraph);
}


int bang_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (BANGEngine*)subgraph->device_graph;
    return engine->BANGEngineRun(subgraph);
}


int bang_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (BANGEngine*)subgraph->device_graph;
    engine->BANGEnginePostRun();
    delete engine;

    return 0;
}


int bang_dev_release(struct device* dev)
{
    (void)dev;
    return 0;
}