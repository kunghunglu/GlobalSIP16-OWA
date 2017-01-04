#include <vector>
#include <utility>
#include <algorithm>

#include <fstream>

#include "caffe/layers/ordered_weighted_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

using namespace std;

namespace caffe {
template <typename Dtype>
void OrderedWeightedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
	    const vector<Blob<Dtype>*>& top) {

	OrderedWeightedParameter ow_param = this->layer_param_.ordered_weighted_param();
  order_ = ow_param.order(); 
  display_ = ow_param.display();
  positive_ = ow_param.positive();

  this->dim_ = bottom[0]->count(ow_param.axis());
  // Initialize and fill the weights;

  this->blobs_.resize(1);
  LOG(INFO)<<"Successful initialize weight!";
  vector<int> weight_shape(2);
  weight_shape[0] = bottom.size();
  weight_shape[1] = bottom[0]->shape(ow_param.axis());
  //LOG(INFO)<<weight_shape[0]<<" "<<weight_shape[1];
  this->blobs_[0] .reset(new Blob<Dtype>(weight_shape));
  
  //shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(ow_param.weight_filler()));
  //weight_filler->Fill(this->blobs_[0].get());
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
  
  // manual weight initialization
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  for(int i=0; i<weight_shape[0]; ++i) {
    for(int j=0; j<weight_shape[1]; ++j) {
      if(i == 0)
        weight[i*this->dim_+j] = 0.2;
      else
        weight[i*this->dim_+j] = weight[(i-1)*this->dim_+j] + 0.2;
    }
  }
  
  /*
  // for debug
  fp <<"Initial weights: "<<weight_shape[0]<<" "<<weight_shape[1]<<endl;
  ofstream fp;
  fp.open("weight.txt", ofstream::out | ofstream::app);
  for(int i=0; i<weight_shape[0]; ++i) {
    for(int j=0; j<weight_shape[1]; ++j) {
      fp<<weight[i*dim_+j]<<" ";
    }
    fp<<endl;
  }
  fp<<"--------------------------------------------------------------------";
  fp.close();
  */
}

template <typename Dtype>
void OrderedWeightedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  for (int i=1; i< bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);

  // initialize the index for remembering relations after sorting
  const int batch_size = bottom[0]->shape(0);
  vector<int> shape_(2);
  shape_[0] = batch_size;
  shape_[1] = this->dim_;
  
  idx_map_.resize(bottom.size());
  for (int i=0; i<idx_map_.size(); ++i) {
    idx_map_[i] .reset(new Blob<Dtype>(shape_));
  }
  
  sort_bottom_.resize(bottom.size());
  for (int i=0; i<sort_bottom_.size(); ++i) {
    sort_bottom_[i] .reset(new Blob<Dtype>(shape_));
  }
  

}

template <typename Dtype>
void OrderedWeightedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = top[0]->count();
  const int batch_size = bottom[0]->shape(0);
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
 
  if(positive_) {
    const int num_weight = this->blobs_[0]->count();
     for(int i=0; i<num_weight; ++i) {
        if(weight[i]<0) {
           weight[i] = 0;
        } 
     }
  }

  const int weight_dim = this->blobs_[0]->shape(1);  
/*
  if (this->phase_ == TEST && display_) {
	  fstream fp;
	  fp.open("weight.txt", ofstream::out | ofstream::app);
	  for(int i=0; i<bottom.size(); ++i) {
	    for(int j=0; j<weight_dim; ++j) {
	      fp<<weight[i*weight_dim+j]<<" ";
	    }
	    fp<<endl;
	  }
	  fp<<"--------------------------------------------------------------------"<<endl;
	  fp.close();
  }
*/
  vector< pair<Dtype, Dtype> > tmp;
  for (int n=0; n<batch_size; ++n){
	  for (int i=0; i<dim_; ++i){
	  	tmp.clear();
	  	for (int j=0; j<bottom.size(); ++j){
	  		const Dtype* bottom_data = bottom[j]->cpu_data();
	  		pair<Dtype, Dtype> item( bottom_data[n*dim_+ i], j);
	  		tmp.push_back(item);
	  	}
	  	sort(tmp.begin(), tmp.end());
      reverse(tmp.begin(), tmp.end());

	  	for (int j=0; j<sort_bottom_.size(); ++j){
	  		Dtype* sort_bottom_data = sort_bottom_[j]->mutable_cpu_data();
	  		Dtype* idx_map_data = idx_map_[j]->mutable_cpu_data();
	  		pair<Dtype, Dtype>& item = tmp[j];
	      sort_bottom_data[n*dim_ + i] = item.first;
	      idx_map_data[n*dim_ + i] = item.second;
	  	}
	  }	
  }
  //  w /dot x
  for(int i=0; i<bottom.size(); ++i) {
  	Dtype* sorted_bottom_data = sort_bottom_[i]->mutable_cpu_data();
  	const Dtype* weight = this->blobs_[0]->cpu_data(); 
    for (int n=0; n < batch_size; ++n) {
      caffe_mul(dim_, sorted_bottom_data + n*dim_, weight+(i*dim_), sorted_bottom_data + n*dim_);
    	//caffe_scal(dim_, weight[i], sorted_bottom_data+n*dim_);
    }
  }

  caffe_set(count, Dtype(0), top_data);
  for (int i = 0; i < bottom.size(); ++i) {
    caffe_axpy(count, Dtype(1), sort_bottom_[i]->cpu_data(), top_data);
  }
  /*
  if(this->phase_ == TRAIN) {
	  LOG(INFO)<<"bottom:"<<endl;
	  for (int i=0; i<bottom.size(); ++i) {
	    const Dtype* data = bottom[i]->cpu_data(); 
	    LOG(INFO)<<data[0]<<" "<<data[1]<<endl;
	  }

	  LOG(INFO)<<"weight:"<<endl;
	  LOG(INFO)<<weight[0]<<" "<<weight[0+1*this->dim_]<<" "<<weight[0+2*this->dim_]<<" "<<weight[0+3*this->dim_]<<endl;
	  LOG(INFO)<<weight[1]<<" "<<weight[1+1*this->dim_]<<" "<<weight[1+2*this->dim_]<<" "<<weight[1+3*this->dim_]<<endl;

	  LOG(INFO)<<"sort bottom:"<<endl;
	  for (int i=0; i<bottom.size(); ++i) {
	    const Dtype* data = sort_bottom_[i]->cpu_data(); 
	    LOG(INFO)<<data[0]<<" "<<data[1]<<endl;
	  }
	  
	  LOG(INFO)<<"top:"<<endl;
	  const Dtype* output = top[0]->cpu_data();
	  LOG(INFO)<<output[0]<<" "<<output[1]<<endl;  
  }
  */
  // for debug
  /*
  if (display_){
	  ofstream fp;
	  fp.open("sort.txt", ofstream::out | ofstream::app);

	  for(int n=0; n<bottom.size(); ++n) {
	  	fp<<"bottom_data:"<<endl;
	    const Dtype* bottom_data = bottom[n]->cpu_data();
	    for(int i=0; i<bottom[0]->shape(0); ++i) {
	  		for(int j=0; j<dim_; ++j) {
	  			fp<<bottom_data[i*dim_+j]<<" ";
	  		}
	  		fp<<endl;
	  	}
	    
	    fp<<"idx_map_:"<<endl;
	    const Dtype* idx_map_data = idx_map_[n]->cpu_data();
	    for(int i=0; i<idx_map_[0]->shape(0); ++i) {
	  		for(int j=0; j<dim_; ++j) {
	  			fp<<idx_map_data[i*dim_+j]<<" ";
	  		}
	  		fp<<endl;
	  	}

	  	fp<<"sorted_bottom_data:"<<endl;
	    const Dtype* sort_bottom_data = sort_bottom_[n]->cpu_data();
	    for(int i=0; i<sort_bottom_[0]->shape(0); ++i) {
	  		for(int j=0; j<dim_; ++j) {
	  			fp<<sort_bottom_data[i*dim_+j]<<" ";
	  		}
	  		fp<<endl;
	  	}  	
	  }

	  fp<<"top_data:"<<endl;
	  for(int i=0; i<top[0]->shape(0); ++i) {
	  	for(int j=0; j<dim_; ++j) {
	  		fp<<top_data[i*dim_+j]<<" ";
	  	}
	  	fp<<endl;
	  }
	  fp<<"------------------------------------------------------"<<endl;
	  fp.close();	
  }
  */
}

template <typename Dtype>
void OrderedWeightedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const int batch_size = bottom[0]->shape(0);
  const Dtype* top_diff = top[0]->cpu_diff();
  
  // gradient w.r.t. bottom data, if necessary.
  for (int i=0; i<bottom.size(); ++i) {
  Dtype* sort_bottom_diff = sort_bottom_[i]->mutable_cpu_diff();
    caffe_copy(count, top_diff, sort_bottom_diff);
  }
  //  w /dot top_diff
  for(int i=0; i<bottom.size(); ++i) {
  	Dtype* sort_bottom_diff= sort_bottom_[i]->mutable_cpu_diff();
  	const Dtype* weight = this->blobs_[0]->cpu_data(); 
    for (int n=0; n < batch_size; ++n) {
      caffe_mul(dim_, sort_bottom_diff + n*dim_, weight+(i*dim_), sort_bottom_diff + n*dim_);
      //caffe_scal(dim_, weight[i], sort_bottom_diff+n*dim_);
    }
  }
  for(int i=0; i<bottom.size(); ++i) {
  	const Dtype* mask = idx_map_[i]->cpu_data();
    for (int j=0; j<idx_map_[0]->count(); ++j) {
    	int index = mask[j];
      Dtype* bottom_diff = bottom[index]->mutable_cpu_diff();
      const Dtype* sort_bottom_diff = sort_bottom_[i]->cpu_diff();
      bottom_diff[j] = sort_bottom_diff[j];
    }
  }

  // gradient w.r.t. weight. Note that we will accumulate diffs.
  
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i=0; i<bottom.size(); ++i) {
  	const Dtype* sort_bottom_data = sort_bottom_[i]->cpu_data();
  	for (int n=0; n<batch_size; ++n) {
  		caffe_mul(dim_, sort_bottom_data+n*dim_, top_diff+n*dim_, weight_diff+i*dim_);
  	}	
  }
  
  /*
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int n=0; n<batch_size; ++n) {
    for (int i=0; i<bottom.size(); ++i) {
    	const Dtype* sort_bottom_data = sort_bottom_[i]->cpu_data();
    	caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, dim_, 1,
    		(Dtype)1., top_diff+n*dim_, sort_bottom_data+n*dim_, (Dtype)1., weight_diff+i);
    }
  }
  */
  //for debug
  /*
  if (display_) {
	  ofstream fp;
	  fp.open("diff.txt", ofstream::out | ofstream::app);
	  fp<<"top_diff:"<<endl;
	  for(int i=0; i<top[0]->shape(0); ++i) {
	  	for(int j=0; j<dim_; ++j) {
	  		fp<<top_diff[i*dim_+j]<<" ";
	  	}
	  	fp<<endl;
	  }
	  fp<<"-------------------"<<endl;
	  fp<<"sorted_bottom_diff:"<<endl;
	  for(int n=0; n<bottom.size(); ++n) {
	    const Dtype* sort_bottom_diff = sort_bottom_[n]->cpu_diff();
	    for(int i=0; i<sort_bottom_[0]->shape(0); ++i) {
	  		for(int j=0; j<dim_; ++j) {
	  			fp<<sort_bottom_diff[i*dim_+j]<<" ";
	  		}
	  		fp<<endl;
	  	}
	  	fp<<"-------------------"<<endl;
	  }

	  fp<<"idx_map_:"<<endl;
	  for(int n=0; n<bottom.size(); ++n) {
	    const Dtype* idx_map_data = idx_map_[n]->cpu_data();
	    for(int i=0; i<idx_map_[0]->shape(0); ++i) {
	  		for(int j=0; j<dim_; ++j) {
	  			fp<<idx_map_data[i*dim_+j]<<" ";
	  		}
	  		fp<<endl;
	  	}
	  	fp<<"-------------------"<<endl;
	  }

	  fp<<"bottom_diff:"<<endl;
	  for(int n=0; n<bottom.size(); ++n) {
	    const Dtype* bottom_diff = bottom[n]->cpu_diff();
	    for(int i=0; i<bottom[0]->shape(0); ++i) {
	  		for(int j=0; j<dim_; ++j) {
	  			fp<<bottom_diff[i*dim_+j]<<" ";
	  		}
	  		fp<<endl;
	  	}
	  	fp<<"-------------------"<<endl;
	  }

	  fp<<"weight_diff:"<<endl;
	  const Dtype* weight_diff = this->blobs_[0]->cpu_diff();
	  const int weight_dim = this->blobs_[0]->shape(1);
	  for(int i=0; i<bottom.size(); ++i) { 
	  	for(int j=0; j<weight_dim; ++j) {
	  		fp<<weight_diff[i*weight_dim+j]<<" ";
	  	}
	  	fp<<endl;
	  }
	  fp<<"------------------------------------------------------"<<endl;
	  for(int i=0; i<this->blobs_[0]->count(); ++i) {
	  	fp<<weight_diff[i]<<" ";
	  }
	  fp<<endl<<"--------------------------"<<endl;
	  fp.close();	
  }
  */
}


INSTANTIATE_CLASS(OrderedWeightedLayer);
REGISTER_LAYER_CLASS(OrderedWeighted);

}  // namespace caffe
