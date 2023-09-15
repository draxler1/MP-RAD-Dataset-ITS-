# MP-RAD-Dataset-ITS
MP-RAD Dataset for the paper, "Detection of Road Accidents using Synthetically Generated Multi-Perspective Accident Videos", published in IEEE Transactions on Intelligent Transportation Systems, 2022.

## Description
Road accidents are often caused by short abnormal events, including traffic violations, abrupt changes in vehicular motion, driver fatigue, etc. Observing an accident event from the proper camera perspective plays a crucial role in detecting accidents. However, capturing such abnormal events from a limited camera perspective may not be possible. We present a deep learning framework to analyze the accident events recorded from multiple perspectives. First, we estimate feature similarity in videos recorded from multiple perspectives. We then divided the video samples into high and low-feature similarity groups. Next, we extract spatio-temporal features from each group using two-branch DCNNs and fuse them using a rank-based weighted average pooling strategy followed by classification. We present a new road accident video dataset (MP-RAD), where each accident event is synthetically generated and captured from five independent camera perspectives using a computer gaming platform.

## Proposed Methodology
![Overview](https://github.com/draxler1/MP-RAD-Dataset-ITS-/assets/49720947/ab69e109-22a9-4909-ad58-fb2b5dcc485f)

## MP-RAD Samples
![MPRAD_Samples](https://github.com/draxler1/MP-RAD-Dataset-ITS-/assets/49720947/8d21eff1-f50e-4f95-9a4f-88e4458a7e46)

## File Structure

- `Video Samples`: Contain a few video samples from MP-RAD.
- `visil/`: Implementation of  ViSiL: Fine-grained spatio-temporal video similarity learning network.
- `Feature_Extraction_WA_Pooling.py`: Extract features, weighted average pooling
- `visil.ipynb`: Notebook for generating feature similarity-based confusion matrix

## Usage
1. Place a single accident video (all five angles) into the query.txt file
2. Obtain the feature similarity-based confusion matrix using `visil.ipynb`
3. Generate rank sequence as per the sorted values
4. Set the `k` value in `Feature_Extraction_WA_Pooling.py`
5. Extract and save video features
6. Construct a classifier network according to the description provided in the Classifier section or see `Classifier.py`
7. Train the network with the provided setting

## Implementation Details
The final classifier model is a 3-layer MLP, where the number of units are 512, 32, and 1, respectively, regularized by dropout with probability of 0.6 between each layer. ReLU and Sigmoid functions are deployed after the first and last layers, respectively. We have trained the model with a learning rate of 0.01 for 350 iterations using Adagrad [43] optimizer.

# **Dataset will be available upon request...**
