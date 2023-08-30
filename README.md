# DUAL-PATH TRANSFORMER BASED NEURAL BEAMFORMER FOR TARGET SPEECH EXTRACTION (DPTBF)
A neural beamformer modeled by a dual-path transformer, bypassing the pre-separation module and intermediate variable estimation.
Shown here are some audio samples from the test set processed by the model.

## Model
We use a dual-path transformer structure to directly predict the beamforming weights from the input feature and noisy covariance matrices, thus avoiding the influence of the estimation accuracy of intermediate variables on the overall performance of the system.
The figure below shows the architecture of our proposed DPTBF.
![main](https://github.com/Aworselife/DPTBF/assets/39001332/4b4bc272-17fc-4a6a-9f78-5b074ea11b1b)
The source code of the model is also provided for understanding the dimension transformation and reproducing the experiment.
The model has no extra tricks, use the [pyroacoustics](https://github.com/LCAV/pyroomacoustics) to generate a data set, and then you can use your own framework for normal training.
