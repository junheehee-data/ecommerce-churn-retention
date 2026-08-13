# ecommerce-churn-retention
# 이커머스 고객 이탈 분석 및 A/B 테스트 기반 리텐션 전략

24만 건의 이커머스 거래 데이터를 바탕으로 단일 변수의 한계를 통계적으로 검증하고, XGBoost 머신러닝 모델을 활용해 복합 이탈 패턴을 도출한 프로젝트입니다. A/B 테스트 샌드박스 설계 및 Incremental Uplift 기반 ROI 산출 모델까지 수립하였습니다.

---

## 1. 프로젝트 개요 (Executive Summary)

* **현황:** 전체 고객 이탈률 **20.05%** (고객 5명 중 1명 이탈 발생).
* **문제점:** 이탈 원인에 대한 정밀 분석 없이 전사 대상 10% 쿠폰 일괄 발송 시, 어차피 남아있을 고객에 대한 할인 잠식(Cannibalization) 및 체리피커 양산으로 연간 약 2.4억 원의 마케팅 예산 낭비 초과.
* **목표:**
  * 통계 검증($t$-test, 카이제곱) 및 머신러닝(XGBoost)을 통한 이탈 핵심 요인 규명
  * 고위험군 타깃 핀셋 CRM 액션 플랜 수립
  * A/B 테스트 기반의 순수 증분(Incremental Uplift) 및 영업이익률 반영 정밀 ROI 산출

---

## 2. 데이터 전처리 및 탐색 (EDA & SQL)

* **데이터 스키마:** 총 248,300건의 고객 거래 및 행동 데이터
* **결측치 처리:** `Returns` 컬럼의 결측치(30,706건)를 미반품(`0`)으로 결측치 임퓨테이션 진행.
* **원-핫 인코딩:** 범주형 변수(`Product Category`, `Payment Method`, `Gender`)에 대해 One-Hot Encoding 적용.

```sql
-- 결제 수단별/반품 여부별 이탈률 기본 집계 SQL
SELECT 
    Payment_Method,
    COUNT(DISTINCT Customer_ID) AS total_customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
FROM raw_data
GROUP BY Payment_Method
ORDER BY churn_rate_pct DESC;
```

---

## 3. 단일 변수 가설 검증 (Statistical Hypothesis Testing)

단일 지표만으로 이탈자를 식별할 수 있는지 알아보기 위해 개별 행동 변수별 통계 검증을 수행했습니다.

| 검증 대상 변수 | 분석 기법 | 통계량 / p-value | 판정 결과 | 비즈니스 해석 |
| :--- | :--- | :--- | :--- | :--- |
| **결제 수단 vs 이탈** | 카이제곱 검증 | $\chi^2 = 11.91, p = 0.0026$ | **유의함 ($p < 0.05$)** | 현금/페이팔(20.25%)이 신용카드(19.66%) 대비 이탈률 높음 |
| **고객 연령 vs 이탈** | 독립표본 $t$-test | $t = -1.15, p = 0.2504$ | 유의하지 않음 | 나이 단독 지표로는 이탈 예측 불가 |
| **총 구매 금액 vs 이탈** | 독립표본 $t$-test | $t = 0.35, p = 0.7242$ | 유의하지 않음 | 구매 금액 단독 지표로는 이탈 예측 불가 |
| **성별 vs 이탈** | 카이제곱 검증 | $p = 0.1709$ | 유의하지 않음 | 단독 성별 차이는 무의미 |
| **구매 수량 vs 이탈** | 카이제곱 검증 | $p = 0.7286$ | 유의하지 않음 | 구매 수량 단독 영향 없음 |

> **Key Insight:** 결제 수단을 제외한 단일 행동 지표만으로는 고객 이탈을 설명하기 어렵습니다. 따라서 변수 간 비선형적 복합 상호작용을 계산할 수 있는 다변량 머신러닝 도입이 필수적임을 증명했습니다.

---

## 4. 다변량 머신러닝 모델링 (XGBoost)

변수 간 복합 상호작용을 포착하기 위해 XGBoost Classifier를 구축하고 주요 변수 중요도(Feature Importance)를 도출했습니다.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Train/Test Split (80:20, Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost 모델 학습
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)
```

### 상위 5개 주요 변수 중요도 (Feature Importance)
1. **`Product Category_Clothing`** (11.16%)
2. **`Customer Age`** (10.22%)
3. **`Gender_Male`** (9.86%)
4. **`Payment Method_Credit Card`** (9.25%)
5. **`Returns (초기 반품 여부)`** (9.20%)

---

## 5. 실행 전략 및 A/B 테스트 프레임워크

### 고위험 타깃 세그먼트 정의
* **조건:** `의류 카테고리 구매` + `현금/페이팔 결제` + `초기 반품(Returns=1) 발생`

### 실행 전략 (Action Plan)
1. **의류 반품 고객 케어:** 반품 완료 즉시 '맞춤 사이즈/핏 가이드' 알림톡 및 무료 교환 바우처 자동 발송.
2. **결제 허들 완화:** 현금/페이팔 이용 고객 대상 '간편결제/카드 등록 시 적립금 3,000원' 지급.

### A/B 테스트 설계 (Sandbox)
* **대상:** 추출된 고위험군 12,000명 (50:50 무작위 할당)
* **처치군 (Group A, 6,000명):** 맞춤 혜택 및 교환 바우처 발송
* **대조군 (Group B, 6,000명):** 기존 일반 메시지 발송 (처치 없음)
* **주요 지표 (Primary Metric):** 30일 이내 재구매율 (Retention Rate)
* **가드레일 지표 (Guardrail Metric):** 마케팅 쿠폰 할인액 대비 영업이익률

---

## 6. 재무적 효과 및 ROI 산출 모델 (Incremental Uplift)

단순 매출 증대가 아닌 **영업이익률(30%)**과 **대조군 대비 순수 증분(Incremental Uplift)**을 반영한 정밀 ROI 수식을 적용했습니다.

### 정밀 성과 산출 공식
$$\text{Incremental Profit} = \Big( N_{\text{target}} \times \Delta_{\text{uplift}} \times \text{AOV} \times \text{Margin Rate} \Big) - \text{Total Campaign Cost}$$

$$\text{Real ROI (\%)} = \frac{\text{Incremental Profit}}{\text{Total Campaign Cost}} \times 100$$

### 시나리오별 예상 성과 (고위험군 12,000명 기준)

| 구분 | 보수적 시나리오 (Conservative) | 목표 시나리오 (Target) |
| :--- | :--- | :--- |
| **타깃 고객 수 ($N_{\text{target}}$)** | 12,000명 | 12,000명 |
| **순수 방어 증분 ($\Delta_{\text{uplift}}$)** | **+4.0%p** (처치 24% vs 대조 20%) | **+6.0%p** (처치 26% vs 대조 20%) |
| **평균 객단가 (AOV)** | 268,000원 | 268,000원 |
| **영업이익률 (Margin Rate)** | **30%** | **30%** |
| **마케팅 집행비 (Cost)** | 18,000,000원 | 18,000,000원 |
| **순이익 (Incremental Profit)** | **20,592,000원** | **39,888,000원** |
| **최종 순 ROI (%)** | **114.4%** | **221.6%** |

---

## 7. 리스크 검토 및 FAQ

* **Q1: 결제 수단별 이탈률 차이(0.59%p)가 작아 보이는데 의미가 있는가?**
  * **A:** 24만 건 이상의 샘플 크기로 카이제곱 검증 결과 $p = 0.0026$으로 통계적 유의성이 명확히 입증되었습니다. 단순 오차가 아닌 결제 단계에서의 UX 허들이 이탈 원인임을 시사합니다.
* **Q2: 의류 카테고리가 이탈 영향도 1위라면 의류 판매를 줄여야 하는가?**
  * **A:** 의류 상품 특성상 반품률이 높아 이탈 촉매 역할을 하는 것입니다. 판매를 줄이는 것이 아니라 교환 프로세스 개선 및 맞춤 사이즈 가이드 제공으로 2차 이탈을 차단하는 접근이 타당합니다.
* **Q3: ROI 수치가 과장되지 않았음을 어떻게 보장하는가?**
  * **A:** 매출(Gross Revenue)이 아닌 영업이익률(30%)을 적용하였고, 어차피 남아있을 고객(Organic Retention)을 제외한 A/B 테스트 순수 증분(Incremental Uplift)만을 성과로 인정하도록 설계하여 리스크를 최소화했습니다.
