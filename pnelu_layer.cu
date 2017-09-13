#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/pnelu_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void PNELUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* slope_data,
    const int div_factor, Dtype t) {

  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : ( pow( (1+(1-t)*in[index]), (1/(1-t)) ) - 1) * slope_data[c];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void PNELUBackward(const int n, const int channels, const int dim,
    const Dtype* out_data, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff,
    const Dtype* slope_data, const int div_factor, Dtype t) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * (pow((out_data[index] + slope_data[c]),t) * pow(slope_data[c], (1-t))) );
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void PNELUParamBackward(const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype t) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ( pow( (1+(1-t)*in_data[index]), (1/(1-t)) ) - 1) * (in_data[index] <= 0);
    for ( int k = 1; k < rows; k++ ) {
        out_diff[index] += in_diff[index + k*rowPitch]
           * ( pow( (1+(1-t)*in_data[index + k*rowPitch]), (1/(1-t)) ) - 1) * (in_data[index + k*rowPitch] <= 0);
        
    }
  }
}

template <typename Dtype>
void PNELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;
  Dtype t = this->layer_param_.pnelu_param().t();
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  PNELUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, slope_data, div_factor, t);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void PNELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  Dtype t = this->layer_param_.pnelu_param().t();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    PNELUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data ,
      backward_buff_.mutable_gpu_diff(), t);
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
       multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        slope_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* slope_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PNELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_data, top_diff, bottom_data, bottom_diff, slope_data,
        div_factor, t);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PNELULayer);


}  // namespace caffe
