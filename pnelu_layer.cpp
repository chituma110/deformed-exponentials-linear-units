#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/pnelu_layer.hpp"

namespace caffe {

#ifdef CPU_ONLY
STUB_GPU(PNELULayer);
#endif

INSTANTIATE_CLASS(PNELULayer);
REGISTER_LAYER_CLASS(PNELU);

}  // namespace caffe
