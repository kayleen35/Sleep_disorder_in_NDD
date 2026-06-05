# Sleep_disorder_in_NDD

# 🦙 Alpa-Ca: 수면장애 기반 신경퇴행성 질환 위험도 예측 서비스

> **공공 의료 빅데이터 × 머신러닝**으로 수면 패턴과 뇌질환의 연관성을 분석하고,  
> 개인 맞춤형 뇌 건강 위험도를 예측하는 Streamlit 기반 웹 서비스

---

## 📌 프로젝트 개요

수면장애(G47 계열)는 알츠하이머(G30)·파킨슨병(G20) 등 신경퇴행성 질환의 **전조 증상**으로 주목받고 있습니다.  
본 프로젝트는 **국민건강보험공단 2023년 진료내역 공공데이터**를 분석하여 그 연관성을 통계적으로 검증하고,  
머신러닝 모델을 통해 개인의 위험도를 예측하는 인터랙티브 서비스를 구현했습니다.

| 항목 | 내용 |
|------|------|
| **데이터** | 국민건강보험공단 진료내역정보 2023 (공공데이터 포털) |
| **대상 질환** | 수면장애(G470–G479) ↔ 알츠하이머(G30), 파킨슨(G20) |
| **분석 규모** | 전국 17개 시도, 18개 연령대, 남녀 구분 |
| **핵심 모델** | EasyEnsemble + XGBoost (Youden Index 최적 임계값) |
| **서비스** | Streamlit 웹 앱 (3페이지 구성) |
| **팀명** | Team 깜졸깜까 |

---

## 🗂️ 레포지토리 구조

```
alpa-ca/
│
├── data/                          # 데이터 파일 (공개 불가 — 아래 안내 참고)
│   ├── 국민건강보험공단_진료내역정보_2023.CSV   # 원본 데이터 (비공개)
│   ├── risk_table.csv             # 수면코드별 위험도 집계 테이블 (모델 입력)
│   └── overall_table.csv          # 전체 통계 요약 테이블 (서비스 통계 탭)
│
├── model/
│   └── ensemble_sleep_model.pkl   # 학습된 EasyEnsemble 모델 + 최적 임계값
│
├── assets/
│   ├── alpacalogo.png             # 서비스 로고
│   └── ggamlogo.png               # 팀 로고
│
├── data_preprocessing.py          # EDA + 통계 분석 + 머신러닝 전체 파이프라인
├── streamlit_service.py           # Streamlit 웹 서비스 메인 앱
├── requirements.txt               # 패키지 의존성
└── README.md
```

> ⚠️ **원본 데이터 안내**: 국민건강보험공단 진료내역 데이터는 공공데이터 포털([data.go.kr](https://www.data.go.kr))에서 직접 신청·다운로드 후 `data/` 경로에 배치해야 합니다. 본 레포에는 저작권 이슈로 원본 데이터를 포함하지 않습니다.

---

## 🔬 분석 파이프라인 (`data_preprocessing.py`)

전체 분석은 아래 흐름으로 진행됩니다. 파일은 Jupyter-style Cell(`# %%`) 구조로 작성되어 있어 VS Code / Jupyter에서 섹션 단위 실행이 가능합니다.

```
[0] 환경 설정 및 패키지 로드
 ↓
[1] 데이터 로드 (국민건강보험공단 원본 CSV, CP949 인코딩)
 ↓
[2] 코드 → 레이블 매핑
    - 연령대코드(1–18), 성별코드, 시도코드, 진료형태, 진료과목
 ↓
[3] EDA (탐색적 데이터 분석)
    - 수면장애 환자 추출 (G47 계열 ICD-10 코드 기반)
    - 알츠하이머(G30) / 파킨슨(G20) 동반 진단 여부 확인
    - 수면세부코드별 발병률 산출
    - 연령·성별·지역별 분포 시각화
    - TableOne 기술통계
 ↓
[4] 통계 분석
    - 수면세부코드별 최초 진단 연령 분포 (히트맵)
    - 오즈비(Odds Ratio) + p-value 산출 (상위 5개 수면코드)
    - 파킨슨 vs 알츠하이머 위험도 비교
    - 처방일수 기반 로지스틱 회귀분석
    - 65세 이상 서브그룹 다변수 로지스틱 회귀
    - Kruskal-Wallis / Mann-Whitney U 검정
 ↓
[5] 머신러닝
    - 특성 공학: 수면세부코드 One-Hot Encoding, 연령 재그룹, 성별 정수화
    - 데이터 불균형 처리: EasyEnsembleClassifier (내부 기본 분류기: XGBoost)
    - 교차검증: Stratified K-Fold (5-Fold)
    - 임계값 최적화: Youden's Index (민감도 + 특이도 합산 최대화)
    - 모델 비교: Random Forest / XGBoost / LightGBM / EasyEnsemble
    - 평가지표: AUC, F1-Score, Accuracy, Sensitivity, Specificity
```

### 📊 주요 분석 결과

| 수면장애 유형 | ICD-10 | 뇌질환 위험도 |
|---|---|---|
| 기타 수면장애 (REM 수면행동장애 포함) | G478 | **가장 높음** |
| 수면무호흡 | G473 | 높음 |
| 발작수면 및 허탈발작 | G474 | 높음 |
| 불면증 | G470 | 중간 |
| 과다수면 | G471 | 중간 |

> REM 수면행동장애(G478 포함)는 파킨슨병·레비소체 치매의 전조 증상으로 알려져 있으며, 본 분석에서도 가장 높은 오즈비를 보였습니다.

---

## 🌐 Streamlit 서비스 (`streamlit_service.py`)

총 **3개 페이지**로 구성된 웹 서비스입니다.

### 화면 구성

| 페이지 | 설명 |
|--------|------|
| **홈** | 서비스 소개 및 예측 페이지 진입 |
| **예측** | 4단계 플로우 기반 개인 뇌 건강 위험도 예측 |
| **통계** | 2023 공공데이터 기반 수면장애 유형별 통계 시각화 |

### 예측 플로우 (4단계)

```
[1단계] 사용자 정보 입력
        - 성별, 생년월일 → 연령대코드 자동 계산
 ↓
[2단계] 수면 증상 체크
        - 7가지 수면 증상 체크박스 (G470–G479 매핑)
 ↓
[3단계] 머신러닝 분석 (로딩)
        - EasyEnsemble 모델 예측 확률 계산
        - 수면코드별 통계 기반 위험도 가중 합산
 ↓
[4단계] 결과 리포트
        - 위험 등급 (정상 / 주의 / 경고 / 위험)
        - Gauge 차트 시각화 (Plotly)
        - 수면 유형별 맞춤 행동 가이드 (AASM·ICSD-3 기반)
```

### 위험도 산출 로직

```python
# 모델 확률(80%) + 통계 기반 위험도(20%) 가중 합산
final_prob = (model_prob * 0.8) + (sleep_score * 0.2)

# 임계값은 학습 시 Youden's Index로 최적화된 값 사용
grade = get_grade(final_prob, best_threshold * 100)
```

---

## ⚙️ 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 및 모델 파일 배치

```
data/
├── risk_table.csv
└── overall_table.csv

model/
└── ensemble_sleep_model.pkl
```

> `data_preprocessing.py`의 전체 파이프라인을 실행하면 위 파일들이 생성됩니다.  
> 원본 데이터는 [공공데이터 포털](https://www.data.go.kr)에서 직접 다운로드해야 합니다.

### 3. Streamlit 앱 실행

```bash
streamlit run streamlit_service.py
```

---

## 📦 주요 패키지

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
lightgbm
imbalanced-learn      # EasyEnsembleClassifier
tableone
streamlit
plotly
joblib
```

---

## 🧠 모델 상세

### EasyEnsemble + XGBoost

- **데이터 불균형 문제**: 전체 수면장애 환자 중 뇌질환 동반 비율이 매우 낮아(< 1%) 일반 분류기는 모두 정상으로 예측하는 편향 발생
- **해결**: EasyEnsembleClassifier로 다수 클래스를 반복 언더샘플링하여 균형 잡힌 앙상블 구성, 내부 기본 분류기로 XGBoost 사용
- **임계값 최적화**: Youden's Index (민감도 + 특이도 - 1)를 최대화하는 지점으로 임계값 설정 → 의료 맥락에서 위양성·위음성 균형 확보
- **모델 저장**: `{'model': ee_model, 'best_threshold': best_threshold}` 딕셔너리로 패키징하여 `.pkl` 저장

### 비교 모델 성능 요약

| 모델 | AUC | F1-Score | 비고 |
|------|-----|----------|------|
| Random Forest | - | - | 정확도 높으나 불균형에 취약 |
| XGBoost | ~0.87 | ~0.12 | AUC 우수, 환자 탐지율 낮음 |
| LightGBM | - | - | 속도 빠름 |
| **EasyEnsemble (최종)** | **최고** | **최고** | 불균형 데이터 최적화 |

---

## ⚠️ 한계점 및 유의사항

- 본 서비스는 **통계적 경향성**에 기반한 참고 도구이며, 의료적 진단을 대체하지 않습니다.
- 공공데이터 특성상 실제 임상 진단 맥락(진단 순서, 발병 선후관계 등)이 반영되지 않습니다.
- 수면장애 세부코드 중 일부(G479 등)는 표본 수가 적어 위험도 추정의 불확실성이 있습니다.
- 증상이 지속되거나 걱정되는 경우 **신경과 전문의 상담**을 받으시기 바랍니다.

---

## 📚 참고 자료

- 미국수면의학회 (AASM) 임상 가이드라인
- 국제 수면장애 분류 3판 (ICSD-3)
- 대한수면학회 / 대한수면의학회 진료 지침
- 국민건강보험공단 공공데이터 포털

---

## 👥 팀 정보

**Team 깜졸깜까** | Service: Alpa-Ca 🦙
