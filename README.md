# SWdacon

SW중심대학 공동 AI 경진대회 2023  
2023.07.03 ~ 2023.07.28 09:59  

# 1. 프로젝트 개요

## 프로젝트 주제
위성 영상, 항공사진 등의 해상도가 향상됨에 따라 판독이 어려웠던 지형, 지물까지 확인할 수 있게 되어 고해상도 원격 탐사 자료를 이용한 다양한 연구가 진행되고 있다. 국토 전역의 건물 객체 추출은 수치지도 갱신, 도시 계획, 3D 건물 모델링, 홍수로 인한 도시 피해, 대략적인 인구 수 예측 등 많은 연구에 활용된다. 건물 객체를 추출하기 위해서는 이미지 내 건물 영역을 분할 할 수 있어야 한다. 하지만 이 작업은 주로 수작업으로 진행되었으며, 많은 시간과 비용이 소모된다는 문제점이 있다.  
따라서 위성 이미지로부터 정밀한 건물의 영역을 분할하여 위와 같은 문제를 해결하기 위해 위성 이미지 건물 영역 분할(Satellite Image Building Area Segmentation) Task를 수행한다.  
위성 이미지의 건물 영역 분할(Image Segmentation)을 수행하는 AI모델을 개발하는 것이 목적이다.  
<p align="center">
  <img src="https://github.com/KURasengan/SWdacon/assets/104672441/23c0fc67-0d1d-4866-9096-e124c3f13743" alt="Untitled (22)" width="700" height="auto">
</p>

## 프로젝트 목표
- 논리적인 모델 선정 및 데이터 처리 방법을 익히고 Pretrained Model 및 Scratch 모델 활용
- High Level Library 활용을 통한 코드 작성 및 고차원적인 이해
- 대회의 흐름을 파악하고 목적을 달성해 리더보드 높은 순위 기록
- Computer Vision의 Segmentation task 이해 및 활용
- 정형 데이터가 아닌 이미지 데이터 활용 및 전처리 역량 획득
- Github, Notion 등 협업 툴 고급 활용 및 협업과 소통 능력 함양

## 활용 장비 및 협업 툴
- 조민서
    - GPU: 2080TI 2대
    - 운영체제: Ubuntu 20.04.6 LTS
- 장원준
    - Kaggle P100, Google Colab T4
- 김영운
    - GPU : TitanX 1대
    - 운영체제 : Ubuntu 20.04 LTS
- 협업툴: Github, Notion

## 프로젝트 구조
```
SWdacon/
│
├── baseline_ver2.4.ipynb - train, inference and make submission code
│
├── EDA/
|   ├── EDA_wonjun.ipynb
|   └── experiement_img_1000.ipynb
│
├── baseline/ - base baseline and baseline code by version
│   ├── baseline.ipynb
│   ├── baseline_ver2.0.ipynb
│   ├── baseline_ver2.1.ipynb
│   ├── baseline_ver2.2.ipynb
│   └── baseline_ver2.3.ipynb
│
├── model/ - Pytorch SMP, FCN8s, TransUNet
│   ├── FCN8s.py
│   ├── smp.py
│   └── transUnet.py
│
├── utils/ - bounding box, kmeans, pixel ensemble, remove shadow, transform and other utils
│   ├── bounding_box_canny_tryouts.ipynb
│   ├── kmeans.py
│   ├── pixel_ensemble.ipynb
│   ├── remove_shadow.py
│   ├── transform.py
│   └── util.py
|
└── visualization/ - visualize image and augmentation tools
    ├── augmentation_test.ipynb
    └── image_viewer.ipynb
```

# 2. 프로젝트 구성 및 역할

- 프로젝트 전반: Baseline 작성, 모델 탐색, Augmentation 실험
- 프로젝트 후반: 메인 모델 선정, Ensemble
    - 장원준: PM, Baseline 작성 및 배포, K-fold Cross Validation, Image Augmentation 실험, Pixel Ensemble, FCN8s 및 Pytorch SMP(UNet, MANET 등) 모델링
    - 조민서: Pytorch SMP 이외의 모델, 방법론 탐색, Loss Function 조합 실험
    - 김영운:Pytorch SMP 모델 실험, 후처리 실험, 모델 성능 평가 시각화
    - 양현서: Pytorch SMP(U++Net 등) 모델 실험

# 3. 프로젝트 결과
**27위 / 968명(227팀)**
![프레젠테이d션1](https://github.com/KURasengan/SWdacon/assets/104672441/623fa63e-e9b0-499d-8852-e0ceb58f641b)

# 5. Contributors
| <img src="https://avatars.githubusercontent.com/u/104672441?v=4" width=250> | <img src="https://avatars.githubusercontent.com/u/120074890?v=4" width=250> | <img src="https://avatars.githubusercontent.com/u/109717248?v=4" width=250> | <img src="https://avatars.githubusercontent.com/u/125948625?v=4" width=250> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [장원준](https://github.com/jwj51720)                                            |                                           [조민서](https://github.com/ChoChoMinSeo)                                            |                                            [김영운](https://github.com/duddns2048)                                            |                                         [양현서](https://github.com/huitsix-86)                                          |
