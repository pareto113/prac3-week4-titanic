# Titanic Survival Prediction — Scaler × Model Baseline 실험

Titanic 데이터셋을 대상으로 스케일러(StandardScaler / MinMaxScaler / NoScaler / MinMax+Robust(Fare))와
모델(KNN / SVM) 조합 8가지를 sklearn Pipeline으로 구성하여 성능을 비교한 실험 프로젝트.

---

## 프로젝트 구조
```
├── titanic.ipynb       # 메인 실험 노트북
├── titanic.csv         # 데이터셋 (Kaggle Titanic)
├── pyproject.toml      # uv 환경 설정
├── uv.lock             # 패키지 버전 고정
└── .gitignore
```

## 설치 및 실행
```bash
git clone https://github.com/pareto113/prac3-week4-titanic.git
cd prac3-week4-titanic
uv sync
uv run jupyter notebook titanic.ipynb
```

> Python 버전 및 패키지 의존성은 `pyproject.toml` / `uv.lock` 에 명시되어 있습니다.

## 실험 구성

| 스케일러 | 모델 | 비고 |
|----------|------|------|
| StandardScaler | KNN / SVM | 기본 조합 |
| MinMaxScaler | KNN / SVM | 기본 조합 |
| NoScaler | KNN / SVM | 전처리 효과 확인용 |
| MinMax+Robust(Fare) | KNN / SVM | Fare 이상치 처리 실험 |

## 주요 결과

- **최고 성능**: MinMax+Robust(Fare) + KNN — Accuracy 0.8268 / F1 0.7891
- Fare 컬럼의 이상치(max 512.3)가 MinMaxScaler 적용 시 성능을 제한하고 있었으며, Fare에만 RobustScaler를 적용하자 F1이 0.031 향상됨
- SVM은 스케일링 미적용 시 F1 0.38로 사실상 작동 불가 — 스케일링이 필수임을 확인