# GlobSIP 16 - Ordered Weighted Averaging Layer

Image Aesthetic Assessment via Deep Semantic Aggregation

## Introducion

We proposed a new parameterized pooling layer - Ordered Weighted Averaging Layer to aggregate features from multi-column networks. We first sort features along specific dimension and multiply them with trainable weights to form a  aggregated feature. The parameters of the network are trained by end-to-end back-propagation technique. Results on the standard benchmark of aesthetic quality assessment shows the effectiveness of our approach.

## Citation

If you find Ordered Weighted Averagin Layer useful in your research, please consider citing:

	Image Aesthetic Assessment via Deep Semantic Aggragation
	Kung-Hung Lu, Kuang-Yu Chang and Chu-Song Chen
	IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016

## Aesthetic quality assessment results:

Performance comparison of different algorithms on AVA dataset. The table shows the accuracy(%) of standard testing set.

|     Method    |  AVA Dataset(%)|
|---------------|----------------|
|Murry et al.   |      68.0      |
|SCNN           |      71.2      |
|AVG-SCNN       |      69.9      |
|DCNN           |      73.3      |
|RDCNN          |      74.5      |
|AlexNet        |      72.3      |
|DMA-Net-ImgFu  |      75.4      |
|**Ours**       |    **78.6**    |

## Installation

Please refer to Caffe [prerequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites)


## How to use ordered weighted layer

**Note**: we are now just providing CPU version, the GPU version is coming soon...

example :

	layer {
	  name: "aggregation"
	  bottom: "1st_pool5"
	  bottom: "2nd_pool5"
	  bottom: "3rd_pool5"
	  bottom: "4th_pool5"
	  type: "OrderedWeighted"
	  top: "aggregation"

	  ordered_weighted_param {
	    OrderOp: DEX              // Sorted in ascending order or descending order
	    positive: true            // Force weights to be positive
	    axis: 1                   // The axis to aggregate
	    weight_filler {           
	      type: "gaussian"
	      mean: 0.5
	      std: 0.1
	    }
	  }
	}


## Resources in this paper

comming soon...

## Contact 

Please feel free to leave suggestions or comments to Kung-Hung Lu (henrylu@iis.sinica.edu.tw), Kuang-Yu Chang (kuangyu@iis.sinica.edu.tw) and Chu-Song Chen (song@iis.sinica.edu.tw)

