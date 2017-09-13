#ifndef CAFFE_PNELU_LAYER_HPP_
#define CAFFE_PNELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {


template <typename Dtype>
class PNELULayer : public NeuronLayer<Dtype> {
 public:
  explicit PNELULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PNELU"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool channel_shared_;
  Dtype t;
  Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
  Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
  Blob<Dtype> bottom_memory_;  // memory for in-place computation
};

}  // namespace caffe

#endif  // CAFFE_PNELU_LAYER_HPP_
