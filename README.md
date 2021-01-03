# Motion-of-Action
Mechanisms of Action (MoA) Prediction - Can you improve the algorithm that classifies drugs based on their biological activity?

In order to solve the problem of multi-class classficiation, the code was refactored for reuse. And when there is non-labeled data, it can be used as 3-stgae.

reference: https://www.kaggle.com/kibuna/kibuna-nn-hs-1024-last-train
[kibuna NN hs:1024 last [TRAIN]](https://www.kaggle.com/kibuna/kibuna-nn-hs-1024-last-train) by [kibuna](https://www.kaggle.com/kibuna)

### Contents
1. Train code
2. Inference code

### What is 3-stage model?
- Stage-1: Feature를 입력받아 nonscore_pred를 학습 및 예측
- Stage-2: Feature + train_nonscore_pred를 입력받아 train-score-pred를 학습 및 예측
- stage-3: train-score-pred를 입력받아 최종 submission 형태로 최종 예측
