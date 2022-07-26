#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "utils.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

#define USE_FP16 //USE_FP16
#define INPUT_NAME "data"
#define OUTPUT_NAME "prob"

using namespace nvinfer1;
static Logger gLogger;

static std::vector<BlockArgs>
	block_args_list = {
		BlockArgs{1, 3, 1, 1, 32, 16, 0.25, true},
		BlockArgs{2, 3, 2, 6, 16, 24, 0.25, true},
		BlockArgs{2, 5, 2, 6, 24, 40, 0.25, true},
		BlockArgs{3, 3, 2, 6, 40, 80, 0.25, true},
		BlockArgs{3, 5, 1, 6, 80, 112, 0.25, true},
		BlockArgs{4, 5, 2, 6, 112, 192, 0.25, true},
		BlockArgs{1, 3, 1, 6, 192, 320, 0.25, true}};


ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt, std::string path_wts, std::vector<BlockArgs> block_args_list, GlobalParams global_params)
{
	float bn_eps = global_params.batch_norm_epsilon;
	DimsHW image_size = DimsHW{global_params.input_h, global_params.input_w};

	std::map<std::string, Weights> weightMap = loadWeights(path_wts);
	Weights emptywts{DataType::kFLOAT, nullptr, 0};
	INetworkDefinition *network = builder->createNetworkV2(0U);
	ITensor *data = network->addInput(INPUT_NAME, dt, Dims3{global_params.in_channels, global_params.input_h, global_params.input_w});
	assert(data);

	int out_channels = roundFilters(32, global_params);
	auto conv_stem = addSamePaddingConv2d(network, weightMap, *data, out_channels, 3, 2, 1, 1, image_size, "_conv_stem");
	auto bn0 = addBatchNorm2d(network, weightMap, *conv_stem->getOutput(0), "_bn0", bn_eps);
	auto swish0 = addSwish(network, *bn0->getOutput(0));
	ITensor *x = swish0->getOutput(0);
	image_size = calculateOutputImageSize(image_size, 2);
	int block_id = 0;
	for (int i = 0; i < block_args_list.size(); i++)
	{
		BlockArgs block_args = block_args_list[i];

		block_args.input_filters = roundFilters(block_args.input_filters, global_params);
		block_args.output_filters = roundFilters(block_args.output_filters, global_params);
		block_args.num_repeat = roundRepeats(block_args.num_repeat, global_params);
		x = MBConvBlock(network, weightMap, *x, "_blocks." + std::to_string(block_id), block_args, global_params, image_size);

		assert(x);
		block_id++;
		image_size = calculateOutputImageSize(image_size, block_args.stride);
		if (block_args.num_repeat > 1)
		{
			block_args.input_filters = block_args.output_filters;
			block_args.stride = 1;
		}
		for (int r = 0; r < block_args.num_repeat - 1; r++)
		{
			x = MBConvBlock(network, weightMap, *x, "_blocks." + std::to_string(block_id), block_args, global_params, image_size);
			block_id++;
		}
	}
	out_channels = roundFilters(1280, global_params);
	auto conv_head = addSamePaddingConv2d(network, weightMap, *x, out_channels, 1, 1, 1, 1, image_size, "_conv_head", false);
	auto bn1 = addBatchNorm2d(network, weightMap, *conv_head->getOutput(0), "_bn1", bn_eps);
	auto swish1 = addSwish(network, *bn1->getOutput(0));
	auto avg_pool = network->addPoolingNd(*swish1->getOutput(0), PoolingType::kAVERAGE, image_size);

	IFullyConnectedLayer *final = network->addFullyConnected(*avg_pool->getOutput(0), global_params.num_classes, weightMap["_fc.weight"], weightMap["_fc.bias"]);
	assert(final);

	final->getOutput(0)->setName(OUTPUT_NAME);
	network->markOutput(*final->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 20);
#ifdef USE_FP16
	config->setFlag(BuilderFlag::kFP16);
#endif
	std::cout << "build engine ..." << std::endl;

	ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
	assert(engine != nullptr);

	std::cout << "build finished" << std::endl;
	// Don't need the network any more
	network->destroy();
	// Release host memory
	for (auto &mem : weightMap)
	{
		free((void *)(mem.second.values));
	}

	return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, std::string wtsPath, std::vector<BlockArgs> block_args_list, GlobalParams global_params)
{
	// Create builder
	IBuilder *builder = createInferBuilder(gLogger);
	IBuilderConfig *config = builder->createBuilderConfig();

	// Create model to populate the network, then set the outputs and create an engine
	ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wtsPath, block_args_list, global_params);
	assert(engine != nullptr);

	// Serialize the engine
	(*modelStream) = engine->serialize();

	// Close everything down
	engine->destroy();
	builder->destroy();
	config->destroy();
}
void doInference(IExecutionContext &context, float *input, float *output, int batchSize, GlobalParams global_params)
{
	const ICudaEngine &engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void *buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(INPUT_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * global_params.input_h * global_params.input_w * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * global_params.num_classes * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * global_params.input_h * global_params.input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * global_params.num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

GlobalParams get_globalParam(std::string backbone, int inc, int h, int w, int num_class) {
	if (backbone =="b0") 
		return GlobalParams{inc, h, w, num_class, 0.001, 1.0, 1.0, 8, -1};
	else if (backbone == "b1")
		return GlobalParams{inc, h, w, num_class, 0.001, 1.0, 1.1, 8, -1};
	else if (backbone == "b2")
		return GlobalParams{inc, h, w, num_class, 0.001, 1.1, 1.2, 8, -1};
	else if (backbone == "b3")
		return GlobalParams{inc, h, w, num_class, 0.001, 1.2, 1.4, 8, -1};
	else if (backbone == "b4")
		return GlobalParams{inc, h, w, num_class, 0.001, 1.4, 1.8, 8, -1};
	else if (backbone == "b5")
		return GlobalParams{inc, h, w, num_class, 0.001, 1.6, 2.2, 8, -1};
	else if (backbone == "b6")
		return GlobalParams{inc, h, w, num_class, 0.001, 1.8, 2.6, 8, -1};
	else if (backbone == "b7")
		return GlobalParams{inc, h, w, num_class, 0.001, 2.0, 3.1, 8, -1};
	else if (backbone == "b8")
		return GlobalParams{inc, h, w, num_class, 0.001, 2.2, 3.6, 8, -1};
	else if (backbone == "l2")
		return GlobalParams{inc, h, w, num_class, 0.001, 4.3, 5.3, 8, -1};
	else
		throw std::runtime_error(std::string("invalid backbone: ") + backbone);
}

bool parse_yaml(std::string yaml_name, int& inc, int& h, int& w, int& class_num, int& batch_size, std::string& backbone){
    YAML::Node config = YAML::LoadFile(yaml_name.c_str());

    // loading optional arguments
    std::cout << "loaded from " << yaml_name << ":" << std::endl;
    if (config["EFFICIENT_NET"]["input_h"]){
        h = config["EFFICIENT_NET"]["input_h"].as<int>();
        std::cout << "input_h: " << h << std::endl;
    }
    if (config["EFFICIENT_NET"]["input_w"]){
        w = config["EFFICIENT_NET"]["input_w"].as<int>();
        std::cout << "input_w: " << w << std::endl;
    }
    if (config["EFFICIENT_NET"]["num_classes"]){
        class_num = config["EFFICIENT_NET"]["num_classes"].as<int>();
        std::cout << "num_classes: " << class_num << std::endl;
    }
    if (config["EFFICIENT_NET"]["batch_size"]){
        batch_size = config["EFFICIENT_NET"]["batch_size"].as<int>();
        std::cout << "batch_size: " << batch_size << std::endl;
    }
	if (config["EFFICIENT_NET"]["in_channels"]){
        inc = config["EFFICIENT_NET"]["in_channels"].as<int>();
        std::cout << "in_channels: " << inc << std::endl;
    }

    // error catching
    if (!config["EFFICIENT_NET"]["backbone"]){
        std::cerr << "Invalid yaml file: failed to find 'EFFICIENT_NET':'backbone'. " << std::endl;
        return false;
    }

    backbone = config["EFFICIENT_NET"]["backbone"].as<std::string>();
    std::cout << "EFFICIENT_NET backbone: " << backbone << std::endl;
    return true;
}

int main(int argc, char **argv)
{
	if (argc!=7 || std::string(argv[1])!="-c" || std::string(argv[3])!="-w" || std::string(argv[5])!="-o") {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./efficientnet -c [.yaml] -w [.wts] -o [.engine]" << std::endl;
        return -1;
    }

	std::string yaml_name(argv[2]);
    int inc,h,w,class_num,batch_size;
    std::string backbone,wts_name(argv[4]),engine_name(argv[6]);
    if (!parse_yaml(yaml_name, inc, h, w, class_num, batch_size, backbone)){
        return -1;
    }

	GlobalParams global_params = get_globalParam(backbone,inc,h,w,class_num);
	// create a model using the API directly and serialize it to a stream
	if (!wts_name.empty())
	{
		IHostMemory *modelStream{nullptr};
		APIToModel(batch_size, &modelStream, wts_name, block_args_list, global_params);
		assert(modelStream != nullptr);

		std::ofstream p(engine_name, std::ios::binary);
		if (!p)
		{
			std::cerr << "could not open plan output file" << std::endl;
			return -1;
		}
		p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
		modelStream->destroy();
		return 0;
	}
	return 0;
}
