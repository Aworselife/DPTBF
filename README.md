# DUAL-PATH TRANSFORMER BASED NEURAL BEAMFORMER FOR TARGET SPEECH EXTRACTION (DPTBF)
A neural beamformer modeled by a dual-path transformer, bypassing the pre-separation module and intermediate variable estimation.
Shown here are some audio samples from the test set processed by the model.

## Model
We use a dual-path transformer structure to directly predict the beamforming weights from the input feature and noisy covariance matrices, thus avoiding the influence of the estimation accuracy of intermediate variables on the overall performance of the system.
The figure below shows the architecture of our proposed DPTBF.
![main](https://github.com/Aworselife/DPTBF/assets/39001332/0cbc9419-10c3-430c-bbe3-a20cccb54bdd)

The source code of the model is provided for understanding the dimension transformation and reproducing the experiment.  
For an introduction to model details, please refer to [Dual-path Transformer Based Neural Beamformer for Target Speech Extraction](https://arxiv.org/abs/2308.15990v2)  
## Train
The model has no extra tricks, use the [pyroacoustics](https://github.com/LCAV/pyroomacoustics) and follow the relevant configuration to generate your own data set, and then you can use your own framework for normal training.
