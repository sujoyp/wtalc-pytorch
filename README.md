# W-TALC: Weakly-supervised Temporal Activity Localization and Classification

## Overview
This package is a PyTorch implementation of the paper [W-TALC: Weakly-supervised Temporal Activity Localization and Classification](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sujoy_Paul_W-TALC_Weakly-supervised_Temporal_ECCV_2018_paper.pdf), by [Sujoy Paul](www.ee.ucr.edu/~supaul/
), Sourya Roy and [Amit K Roy-Chowdhury](http://www.ee.ucr.edu/~amitrc/) and published at [ECCV 2018](https://eccv2018.org/). The TensorFlow implementation can be found [here](https://github.com/sujoyp/wtalc-tensorflow).

## Dependencies
This package uses or depends on the the following packages:
1. PyTorch 0.4.1, Tensorboard Logger 0.1.0
2. Python 3.6
3. numpy, scipy among others

## Data
The Kinetics pre-trained I3D features for Thumos14 dataset can be downloaded [here](https://emailucr-my.sharepoint.com/:f:/g/personal/sujoy_paul_email_ucr_edu/Es1zbHQY4PxKhUkdgvWHtU0BK-_yugaSjXK84kWsB0XD0w?e=I836Fl). The annotations are included with this package. 

## Running
This code can be run using two diferent datasets - Thumos14 and Thumos14reduced. The later dataset contain only the data points which has temporal boundaries. The dataset name (with other parameters can be changed in options.py). The file to be executed is main.py. The results can be viewed using tensorboard logger or the text file named .log generated during execution.

## Citation
Please cite the following work if you use this package.
```javascript
@inproceedings{paul2018w,
  title={W-TALC: Weakly-supervised Temporal Activity Localization and Classification},
  author={Paul, Sujoy and Roy, Sourya and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={563--579},
  year={2018}
}
```

## Contact 
Please contact the first author of the associated paper - Sujoy Paul (supaul@ece.ucr.edu) for any further queries.


