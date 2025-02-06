# Pipeline을 활용한 머신러닝 프로젝트

이 프로젝트는 scikit-learn의 Pipeline을 활용하여 데이터 전처리부터 모델 학습까지의 과정을 자동화하는 방법을 보여줍니다.

## 프로젝트 개요

- 보험 청구 금액을 예측하는 머신러닝 모델 개발
- Pipeline을 사용하여 데이터 전처리와 모델링 과정을 자동화

## Pipeline 구성 방법

### 1. 전처리기 준비

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 수치형 데이터 처리
numeric_columns = ['Age', 'BMI', 'NumVisits']
imputer = SimpleImputer()

# 레이블 인코딩할 컬럼
label_columns = ['Gender', 'Smoker']
label_encoder = OrdinalEncoder()

# 원핫 인코딩할 컬럼
onehot_columns = ['Region']
onehot_encoder = OneHotEncoder()

# ColumnTransformer로 전처리기 결합
preprocessor = ColumnTransformer([
    ('num', imputer, numeric_columns),
    ('label', label_encoder, label_columns),
    ('onehot', onehot_encoder, onehot_columns)
])
```

### 2. Pipeline 생성

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('modeling', RandomForestRegressor(random_state=42))
])
```

### 3. 모델 학습

```python
pipeline.fit(X_train, y_train)
```

### 4. 예측

```python
y_pred = pipeline.predict(X_test)
```

## Pipeline 사용의 장점

1. **코드 간소화**: 여러 전처리 단계와 모델링을 하나의 객체로 관리
2. **재현성 보장**: 동일한 전처리 과정을 새로운 데이터에 자동 적용
3. **배포 용이성**: 하나의 파일로 저장하여 서비스에 쉽게 배포 가능

## 모델 저장 및 불러오기

```python
import joblib

# 모델 저장
joblib.dump(pipeline, 'pipeline.pkl')

# 모델 불러오기
loaded_pipeline = joblib.load('pipeline.pkl')
```

## 프로젝트 결과

- MSE (Mean Squared Error): 46.85
- R² Score: 0.75




