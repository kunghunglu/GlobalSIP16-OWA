#ifndef CAFFE_ORDERED_WEIGHTED_LAYER_HPP_
#define CAFFE_ORDERED_WEIGHTED_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OrderedWeightedLayer : public Layer<Dtype> {
  public:
    explicit OrderedWeightedLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {return "Ordered Weighted"; }
    virtual inline int MinBottomBlobs() const {return 1;}
    virtual inline int ExactNumTopBlobs() const {return 1;}

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    
   OrderedWeightedParameter_OrderOp order_;
   vector<shared_ptr<Blob<Dtype> > > idx_map_;
   vector<shared_ptr<Blob<Dtype> > > sort_bottom_;
   //Blob<int> idx_;
   int dim_;
   bool display_;
   bool positive_;
};

} //namespace caffe


#endif //CAFFE_ORDERED_WEIGHTER_LAYER_HPP_
