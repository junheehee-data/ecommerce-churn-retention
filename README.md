# ecommerce-churn-retention
# 이커머스 고객 이탈 분석 및 A/B 테스트 기반 리텐션 전략

24만 건의 이커머스 거래 데이터를 활용하여 고객 이탈 원인을 규명하고, 타깃 리텐션 프로모션 전략 및 성과 측정 모델(ROI)을 수립한 프로젝트입니다.

---

## 1. 프로젝트 개요

* **배경:** 전체 고객 이탈률 20.05% 기록. 이탈 원인 파악 없이 진행되는 전체 쿠폰 발송 중심의 마케팅으로 인해 예산 효율성 저하.
* **목표:**
  * 이탈에 영향을 미치는 핵심 변수 및 복합 패턴 규명 (통계 검증 & 머신러닝)
  * 고위험군 타깃 세그먼트 정의 및 핀셋 리텐션 액션 플랜 수립
  * A/B 테스트 기반의 정밀 성과(Incremental Uplift) 측정 모델 구축

---

## 2. 데이터 전처리 및 탐색

* **결측치 처리:** `Returns` 컬럼의 결측치(약 3만 건)를 미반품(`0`) 데이터로 임퓨테이션.
* **인코딩:** 범주형 변수(`Product Category`, `Payment Method`, `Gender`) One-Hot Encoding 적용.
* **데이터 탐색:** SQL 기반 집계 결과, 결제 수단 및 상품 카테고리별 이탈률에 유의미한 수치 차이 포착.

```sql
-- 결제 수단별 이탈률 집계
SELECT 
    Payment_Method,
    COUNT(DISTINCT Customer_ID) AS total_customers,
    SUM(Churn) AS churned_customers,
    ROUND(AVG(Churn) * 100, 2) AS churn_rate_pct
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    eval_metric='logloss', 
    random_state=42
)
model.fit(X_train, y_train)
FROM raw_data
GROUP BY Payment_Method
ORDER BY churn_rate_pct DESC;
