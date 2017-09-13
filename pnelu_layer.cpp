#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/pnelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void PNELULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PNELUParameter pnelu_param = this->layer_param().pnelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = pnelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (pnelu_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(pnelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void PNELULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void PNELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  Dtype t = this->layer_param_.pnelu_param().t();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  float total_bottom = 0.0;
  float total_bottom_neg = 0.0;
  int num_bottom_neg = 0;
  int num_bottom_pos = 0;
  float total_bottom_pos = 0.0;
  float total_top = 0.0;
  float total_top_neg = 0.0;
  float total_top_pos = 0.0;
  int num_top_neg = 0;
  int num_top_pos = 0;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    total_bottom += bottom_data[i];
    if(bottom_data[i] >0){
        total_bottom_pos +=bottom_data[i];
 		num_bottom_pos += 1;
    }else{
		total_bottom_neg +=bottom_data[i];
		num_bottom_neg += 1;
    } 
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + slope_data[c] * ( pow( std::max(Dtype(0),(Dtype(1)+(Dtype(1)-t)*(std::min(bottom_data[i], Dtype(0))))), Dtype(1)/(Dtype(1)-t) ) - Dtype(1) );
    
    total_top += top_data[i];
	if(top_data[i] > 0){
		total_top_pos += top_data[i]; 
		num_top_pos += 1;
    }else{
		total_top_neg += top_data[i]; 
		num_top_neg +=1;
    }
 
  }
  LOG(INFO) << "the mean_bottom is: " << total_bottom / count << "\n";
  LOG(INFO) << "the mean_bottom_pos is: " << total_bottom_pos / num_bottom_pos << "\n";
  LOG(INFO) << "the mean_bottom_neg is: " << total_bottom_neg / num_bottom_neg << "\n";
  LOG(INFO) << "the mean_top is: " << total_top / count << "\n";
  LOG(INFO) << "the mean_top_pos is: " << total_top_pos / num_top_pos << "\n";
  LOG(INFO) << "the mean_top_neg is: " << total_top_neg / num_top_neg << "\n";
}

template <typename Dtype>
void PNELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  Dtype t = this->layer_param_.pnelu_param().t();
  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      slope_diff[c] += top_diff[i] * ( pow((Dtype(1)+(Dtype(1)-t)*(std::min(bottom_data[i], Dtype(0)))), Dtype(1)/(Dtype(1)-t) ) - Dtype(1) ) * (bottom_data[i] <= 0);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + (pow((slope_data[c] + top_data[i]),t) * pow(slope_data[c], (Dtype(1)-t))) * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PNELULayer);
#endif

INSTANTIATE_CLASS(PNELULayer);
REGISTER_LAYER_CLASS(PNELU);

}  // namespace caffe
