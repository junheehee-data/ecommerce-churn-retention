# ecommerce-churn-retention
# 🛒 [E-commerce] 고객 이탈 원인 규명 및 A/B 테스트 기반 Retention Strategy
> **24만 건 거래 데이터 기반 고객 이탈 요인 분석, XGBoost 머신러닝 구축, A/B 테스트 및 Incremental ROI 모델링**

---

## 📌 Executive Summary

본 프로젝트는 24만 건의 이커머스 거래 데이터를 기반으로 고객 이탈의 핵심 요인을 규명하고, 데이터 기반의 핀셋 CRM 전략 및 **A/B 테스트 사전 성과 측정 프레임워크(Pre-experiment Framework)**를 수립한 프로젝트입니다.

단일 행동 변수의 한계를 통계적 가설 검증($t$-test, $\chi^2$)으로 파악한 후 **XGBoost 모델링**을 통해 비선형 다변량 이탈 패턴을 포착했습니다. 이를 바탕으로 **Incremental Uplift**와 **영업이익률(30%)**을 반영한 정밀 ROI 추정 모델을 구축하여 **최대 221.6%의 Net ROI 창출 시나리오**를 제시합니다.

---

## 🛠️ Tech Stack & Directory Structure

* **Data Analysis & SQL:** BigQuery / MySQL (Complex Join, Window Function, Aggregation)
* **Programming & ML:** Python (Pandas, NumPy, SciPy, XGBoost, Scikit-learn)
* **Statistical Testing:** Independent $t$-test, Chi-square Test of Independence ($\chi^2$)
* **Experiment Design:** Pre-experiment A/B Testing Sandbox, Incremental Uplift ROI Valuation

```text
ecommerce-churn-retention/
├── README.md
├── requirements.txt
├── sql/
│   └── data_analysis.sql
└── src/
    └── churn_pipeline.py
```

---

## 1. 프로젝트 개요 (Problem Definition & Objectives)

* **문제 정의:** 전체 고객 이탈률 **20.05%** 달성. 이탈 원인에 대한 정밀한 정의 없이 시행되는 전사 대상 10% 할인 쿠폰 일괄 발송으로 인해, 유지 가능 고객에 대한 **할인 잠식(Cannibalization)** 및 연간 약 2.4억 원 규모의 마케팅 예산 비효율 발생.
* **핵심 목표:**
  1. 통계 검증 및 머신러닝을 활용한 고위험 이탈 요인 규명
  2. 고위험군 타깃 핀셋 CRM 액션 플랜 수립
  3. A/B 테스트 기반의 순수 증분(Incremental Uplift) 및 영업이익률 반영 정밀 ROI 측정 프레임워크 구축

---

## 2. 데이터 전처리 및 탐색 (EDA & SQL)

* **데이터 규모:** 총 248,300건의 고객 거래 및 행동 로그
* **결측치 임퓨테이션:** `Returns` 컬럼의 결측치(30,706건) 분석 후 미반품(`0`) 상태로 전처리
* **특성 인코딩:** 범주형 변수(`Product Category`, `Payment Method`, `Gender`) 대상 One-Hot Encoding 적용

```sql
-- 결제 수단별/반품 여부별 이탈률 기본 집계 Query
SELECT 
    Payment_Method,
    COUNT(DISTINCT Customer_ID) AS total_customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
FROM cleaned_ecommerce_churn
GROUP BY Payment_Method
ORDER BY churn_rate_pct DESC;
```

---

## 3. 단일 변수 가설 검증 (Statistical Testing)

개별 행동 변수가 이탈에 직접적인 영향을 미치는지 통계적 유의성을 검증했습니다.

| 검증 변수 | 적용 기법 | 통계량 및 p-value | 판정 | 비즈니스 해석 |
| :--- | :--- | :--- | :---: | :--- |
| **결제 수단** | 카이제곱 검증 | $\chi^2 = 11.91, p = 0.0026$ | **유의** | 현금/페이팔(20.25%)이 카드(19.66%) 대비 결제 허들로 인한 이탈 높음 |
| **고객 연령** | 독립표본 $t$-test | $t = -1.15, p = 0.2504$ | 기각 | 연령 단독 지표로는 이탈 예측 불가 |
| **총 구매 금액** | 독립표본 $t$-test | $t = 0.35, p = 0.7242$ | 기각 | 구매 금액 단독 지표로는 이탈 예측 불가 |
| **성별** | 카이제곱 검증 | $p = 0.1709$ | 기각 | 성별 단독 차이는 통계적으로 우연일 가능성 높음 |
| **구매 수량** | 카이제곱 검증 | $p = 0.7286$ | 기각 | 단발성/다량 구매 여부 단독 영향 없음 |

> **Key Takeaway:** 결제 수단을 제외한 단일 변수만으로는 이탈을 유의미하게 설명하기 어렵습니다. 변수 간 비선형 복합 상호작용을 포착하는 다변량 머신러닝(XGBoost) 도입의 당위성을 통계적으로 입증했습니다.

---

## 4. 다변량 머신러닝 모델링 (XGBoost)

변수 간 복합 상호작용을 포착하기 위해 **XGBoost Classifier**를 구축하고 주요 변수 중요도(Feature Importance)를 도출했습니다.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Stratified Train/Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost 모델 학습
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)
```

### 📊 Model Performance
* **AUC-ROC Score:** `0.764`
* **F1-Score (이탈 클래스 기준):** `0.69` (Precision: 0.71 / Recall: 0.68)
* **Accuracy:** `81.2%`

### 🔑 Top 5 Feature Importance
1. **`Product Category_Clothing`** — **11.16%**
2. **`Customer Age`** — **10.22%**
3. **`Gender_Male`** — **9.86%**
4. **`Payment Method_Credit Card`** — **9.25%**
5. **`Returns (초기 반품 경험)`** — **9.20%**

---

## 5. CRM 액션 플랜 & A/B 테스트 프레임워크

### Target Segment
* **고위험군 조건:** `의류 카테고리 구매` + `현금/페이팔 결제` + `초기 반품(Returns=1) 경험` (12,000명)

### Action Plan
1. **반품 고객 케어:** 반품 완료 즉시 '맞춤 사이즈/핏 가이드' 알림톡 및 무료 교환 바우처 발송 (2차 이탈 차단)
2. **결제 허들 완화:** 현금/페이팔 이용 고객 대상 '간편결제 등록 시 적립금 3,000원' 지급 (결제 UX 개선)

### A/B Test Design (Sandbox)
* **대상:** 고위험 타깃군 12,000명 (50:50 무작위 할당)
* **처치군 (Group A, 6,000명):** 맞춤형 케어 메시지 및 바우처 지급
* **대조군 (Group B, 6,000명):** 기존 일반 메시지 발송 (Control)
* **Primary Metric:** 30일 이내 재구매율 (Retention Rate)
* **Guardrail Metric:** 마케팅 집행비 대비 영업이익률 (Profit Margin)

---

## 6. 재무적 성과 및 ROI 모델 (Incremental Uplift)

매출 전체를 성과로 왜곡하는 오류를 차단하기 위해 **영업이익률(30%)**과 **대조군 대비 순수 증분(Incremental Uplift)** 지표를 반영한 ROI 수식을 설계했습니다.

### Performance Valuation Formula
$$\text{Incremental Profit} = \Big( N_{\text{target}} \times \Delta_{\text{uplift}} \times \text{AOV} \times \text{Margin Rate (30\%)} \Big) - \text{Total Campaign Cost}$$

$$\text{Real Net ROI (\%)} = \frac{\text{Incremental Profit}}{\text{Total Campaign Cost}} \times 100$$

### Scenario Analysis (Target: 12,000명)

| 항목 (Metric) | Conservative Scenario | Target Scenario |
| :--- | :---: | :---: |
| **Target Population ($N_{\text{target}}$)** | 12,000명 | 12,000명 |
| **Incremental Uplift ($\Delta_{\text{uplift}}$)** | **+4.0%p** (처치 24% vs 대조 20%) | **+6.0%p** (처치 26% vs 대조 20%) |
| **Average Order Value (AOV)** | 268,000원 | 268,000원 |
| **Margin Rate** | **30%** *(패션/커머스 표준)* | **30%** *(패션/커머스 표준)* |
| **Campaign Cost** | 18,000,000원 | 18,000,000원 |
| **Incremental Profit (순이익)** | **20,592,000원** | **39,888,000원** |
| **Net ROI (%)** | **114.4%** | **221.6%** |

---

## 7. 리스크 검토 및 FAQ

* **Q1. 결제 수단별 이탈률 차이(0.59%p)의 비즈니스적 유의미성**
  * **A:** 24만 건 이상의 대용량 샘플에서 카이제곱 검증 결과 $p = 0.0026$으로 통계적 유의성이 명확히 검증되었습니다. 이는 단순 오차가 아닌 결제 UX 상의 허들이 실질적 이탈을 유발함을 의미합니다.
* **Q2. 의류 카테고리의 이탈 중요도 1위에 대한 전략적 해석**
  * **A:** 의류는 품목 특성상 사이즈 및 재질 불만족으로 인한 반품률이 높아 이탈의 촉매로 작동합니다. 의류 판매 축소가 아닌, 교환 프로세스 개선 및 핏 가이드 제공을 통해 2차 이탈을 차단하는 접근이 타당합니다.
* **Q3. ROI 산출 모델의 신뢰성 보장 방안**
  * **A:** 매출(Gross Revenue)이 아닌 영업이익률(30%)을 기준으로 계산하였으며, 자연 유지(Organic Retention) 수치를 제외한 A/B 테스트 순수 증분(Incremental Uplift)만을 성과로 인정하여 재무적 리스크를 최소화했습니다.

---

## 8. 한계점 및 향후 과제 (Limitations & Future Scope)

1. **사전 시뮬레이션 한계:** 본 분석은 거래 로그 기반의 사전 실험 설계(Pre-experiment Framework)로, 실제 라이브 집행 시 외부 경제 상황 및 경쟁사 프로모션 변수의 추가 통제가 요구됩니다.
2. **계절성(Seasonality) 미반영:** 단년도 데이터 특성상 FW/SS 시즌별 이탈 변동성을 반영하지 못했으므로, 시계열 알고리즘(Prophet 등)을 결합한 고도화가 요구됩니다.
3. **LTV 관점 확장:** 30일 단기 재구매 방어 외에, 마케팅 액션이 장기 고객 생애 가치(LTV)에 미치는 영향을 추적하는 장기 평가 시스템 구축이 필요합니다.

---

## 🚀 How to Run

1. **의존성 라이브러리 설치**
   ```bash
   pip install -r requirements.txt
   ```
2. **파이프라인 실행 (통계 검증, ML 학습, ROI 시뮬레이션 원스톱 실행)**
   ```bash
   python src/churn_pipeline.py
   ```
