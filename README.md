# 20_Motion-of-Action

이 github repository는 가짜연구소 1기 대회참가팀(신약개발)의 스터디 결과물입니다.
Mechanisms of Action (MoA) Prediction - Can you improve the algorithm that classifies drugs based on their biological activity?
In order to solve the problem of multi-class classficiation, the code was refactored for reuse. And when there is non-labeled data, it can be used as 3-stgae.


### Contents
1. [Gitbook](https://app.gitbook.com/@pseudo-lab/s/1st-moa/)
    1. Paper review
        1. [CPEM: Accurate cancer type classification based on somatic alterations using an ensemble of a random](https://app.gitbook.com/@pseudo-lab/s/1st-moa/paper-review/cpem-accurate-cancer-type-classification-based-on-somatic-alterations-using-an-ensemble-of-a-random)
        2. TabNet
    2. Notebook review
        1. [신약의 작용 메커니즘 예측 | 차근차근 완벽한 설명 | EDA에서 앙상블까지](https://app.gitbook.com/@pseudo-lab/s/1st-moa/notebook-review/moa-public-26th-solution)
        2. [Stacking & Blending](https://app.gitbook.com/@pseudo-lab/s/1st-moa/notebook-review/untitled)
        3. [MoA Inference 노트북 작성](https://pseudo-lab.gitbook.io/1st-moa/notebook-review/moa-inference)
        4. [5-Fold train/valid set](https://app.gitbook.com/@pseudo-lab/s/1st-moa/~/drafts/-MQBuOymvry30KmP0Geo/notebook-review/untitled-1/@drafts)
        5. [SMOTE - Upsampling](https://app.gitbook.com/@pseudo-lab/s/1st-moa/notebook-review/nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp)
        6. [Catboost Training](https://pseudo-lab.gitbook.io/1st-moa/notebook-review/catboost-training)
    3. Solution review
        1. [1st Place Winning Solution - Hungry for Gold](https://pseudo-lab.gitbook.io/1st-moa/solution-review/untitled-1)
        2. [3rd Place Public - We Should Have Trusted CV - 118th Private](https://pseudo-lab.gitbook.io/1st-moa/solution-review/untitled-2)
        3. 5th solution
        4. 7th solution
        5. 8th solution
        6. [14th-solution](https://pseudo-lab.gitbook.io/1st-moa/solution-review/14th-solution)
        7. [Public 46th / Private 34th Solution](https://pseudo-lab.gitbook.io/1st-moa/solution-review/public-46th-private-34th-solution)
2. 3-stage Model (Train/Inference code)
3. TabNet (Train/Inference code)
4. Ensemble


### What is 3-stage model?
- Stage-1: Feature를 입력받아 nonscore_pred를 학습 및 예측
- Stage-2: Feature + train_nonscore_pred를 입력받아 train-score-pred를 학습 및 예측
- stage-3: train-score-pred를 입력받아 최종 submission 형태로 최종 예측


### Reference
- [MoA / PyTorch NN starter](https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter)
- [kibuna NN hs:1024 last [TRAIN]](https://www.kaggle.com/kibuna/kibuna-nn-hs-1024-last-train) by [kibuna](https://www.kaggle.com/kibuna)
- https://github.com/trent-b/iterative-stratification
- https://www.kaggle.com/konradb/anomaly-detection
- https://people.csail.mit.edu/kalyan/AI2_Paper.pdf
- [MLSMOTE + MoA | Pytorch | 0.01859 | RankGauss | PCA | NN (0.01859)](https://www.kaggle.com/tolgadincer/mlsmotehttps://www.kaggle.com/kushal1506/moa-pytorch-0-01859-rankgauss-pca-nn)
- [R based Keras best model (0.01842) Fork of 2heads... looper super puper](https://www.kaggle.com/demetrypascal/fork-of-2heads-looper-super-puper)
- [Tabnet (0.01854)](https://www.kaggle.com/hiramcho/moa-predictions-overfitting-with-tabnet)
- MLSMOTE + MoA: Multi Input ResNet Model
- https://www.kaggle.com/hwigeon/nn-svm-tabnet-xgb-with-pca-cnn-stacking-without-pp
