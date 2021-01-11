# 20_Motion-of-Action

![PseudoLab](https://img.shields.io/badge/Community-PseudoLab-ff69b4.svg)
![Since](https://img.shields.io/badge/Since-2020-blueviolet.svg)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Kaggle](https://img.shields.io/badge/Interest-Kaggle-9cf.svg)

이 github repository는 가짜연구소 1기 대회참가팀(신약개발)의 스터디 결과물입니다.

Mechanisms of Action (MoA) Prediction - Can you improve the algorithm that classifies drugs based on their biological activity?
In order to solve the problem of multi-class classficiation, the code was refactored for reuse. And when there is non-labeled data, it can be used as 3-stgae.

### Contributors
- [김찬란](https://github.com/seriousran), 문성채, [박원준](https://github.com/WonJunPark), 오휘건, [이현수](https://github.com/soomiles)


### Contents
1. [Gitbook](https://app.gitbook.com/@pseudo-lab/s/1st-moa/)
    1. Paper review
        1. [CPEM: Accurate cancer type classification based on somatic alterations using an ensemble of a random](https://app.gitbook.com/@pseudo-lab/s/1st-moa/paper-review/cpem-accurate-cancer-type-classification-based-on-somatic-alterations-using-an-ensemble-of-a-random)
        2. [TabNet](https://pseudo-lab.com/TabNet-9336b1657d524445bed38538dc1368d2)
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
2. [3-stage Model](https://github.com/Pseudo-Lab/20_MoA/tree/main/3_stage_model)
3. [TabNet (Train/Inference code)](https://github.com/Pseudo-Lab/20_MoA/tree/main/tabnet)
4. [Ensemble](https://github.com/Pseudo-Lab/20_MoA/blob/main/ensemble/Code%20for%20Blending.ipynb)


### What is 3-stage model?
- Stage-1: Feature를 입력받아 nonscore_pred를 학습 및 예측
- Stage-2: Feature + train_nonscore_pred를 입력받아 train-score-pred를 학습 및 예측
- stage-3: train-score-pred를 입력받아 최종 submission 형태로 최종 예측


### 스터디 리뷰
1. 이 스터디를 통해서 배운 것!
    - Theory
        - Multi-label Classification
        - CPEM paper review → 의학쪽에서 사용되는 머신러닝 접근 방법 옅보기
    - Feature Engineering
        - MLSMOTE
        - Variance Threshold
    - Model
        - TabNet
        - 2-head, 3-head ResNet - head를 여러개 사용하는 모델 구조에 대해서 익힘
        - 3-stage model을 활용한 non-scored data 학습 및 추론
        - **작고 불균형이 심한 다중 레이블 데이터셋에서 과적합의 위험을 극복하기 위해** 모델 학습 프로세스의 정규화 방법으로 **레이블 평활화(label smoothing)** 및 **가중치 감소(weight decay)**를 적용
    - Ensemble & Stacking
2. 후기
    - 찬란: Multi-label Classfication과 3-stage 딥러닝 모델을 활용하여 robust한 성능을 낸 모델에 대해서 학습한게 좋았습니다. 특히 다른분들을 통해서 제대로 이해하지 않고 넘어갔던 부분들에 대해 다시 한번 생각해보게 되고, 논의함으로 깊이 이해할 수 있는게 좋았습니다. '신약개발' 대회 참가라는 타이틀이었는데, '신약개발'에 대한 내용 스터디 부족이 아쉬움!
    - 원준: 2head resnet사용 방법과 stacking & blending 사용방법에 대해 학습한 부분이 좋았습니다. 하지만 끝까지 도메인 파악이 부족했던것 같지만 메달을 받을 수 있어 만족합니다.
    - 휘건: public leader board 점수에 연연하지 않고 CV 점수를 잘 따라가면 된다고 배웠습니다. 적당한 성능의 모델을 많이 잘 앙상블하는게 효과가 많이 좋은 것 같습니다.
    - 성채: 새로운 데이타 전처리 분석 방법론들을 배웠습니다.
3. 다음에는 어떻게 접근할지!
    - Factor Analysis, UMAP을 활용한 Feature Engineering
    - Statistical Feature Engineering
        - high correlation feature → product해서 새로운 feature 생성
    - Local 장비를 이용해 학습 진행
    - 앙상블하기 위한 노력을 일찍!
    - [Adversarial validation](https://towardsdatascience.com/adversarial-validation-ca69303543cd)


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
