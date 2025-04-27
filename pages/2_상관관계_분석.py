import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import get_property_data, get_population_data, get_finance_data

st.set_page_config(page_title="상관관계 분석", page_icon="🔄")

st.title("상관관계 분석")
st.write("토지가격과 인구밀집도 간의 상관관계를 분석합니다.")

# 사이드바 - 지역 선택
st.sidebar.header("데이터 선택")
region = st.sidebar.selectbox(
    "지역 선택",
    ["전체", "서초구", "영등포구", "중구"]
)

if region == "전체":
    region = None

# 데이터 로드
property_data = get_property_data(region=region)
population_data = get_population_data(region=region)

# 컬럼명 확인
if 'SGG' in property_data.columns:
    property_district_col = 'SGG'
    property_price_col = 'MEME_PRICE_PER_SUPPLY_PYEONG'
    property_jeonse_col = 'JEONSE_PRICE_PER_SUPPLY_PYEONG'
    property_date_col = 'YYYYMMDD'
else:
    property_district_col = '시군구명'
    property_price_col = '매매가격(만원/평)'
    property_jeonse_col = '전세가격(만원/평)'
    property_date_col = '날짜'
    
if 'DISTRICT_CODE' in population_data.columns:
    pop_district_col = 'DISTRICT_CODE'
    pop_resident_col = 'RESIDENTIAL_POPULATION'
    pop_visit_col = 'VISITING_POPULATION'
    pop_work_col = 'WORKING_POPULATION'
    pop_date_col = 'STANDARD_YEAR_MONTH'
else:
    pop_district_col = '지역코드'
    pop_resident_col = '거주인구'
    pop_visit_col = '방문인구'
    pop_work_col = '근무인구'
    pop_date_col = '기준연월'

# 데이터 전처리
st.subheader("데이터 전처리")
st.write("상관관계 분석을 위해 데이터를 결합합니다.")

# 날짜 형식 변환
property_data['년월'] = pd.to_datetime(property_data[property_date_col]).dt.strftime('%Y%m')
population_data['년월'] = population_data[pop_date_col]

# 데이터 집계
property_monthly = property_data.groupby([property_district_col, '년월']).agg({
    property_price_col: 'mean',
    property_jeonse_col: 'mean'
}).reset_index()

# 지역코드 매핑
region_mapping = {'11650': '서초구', '11560': '영등포구', '11140': '중구'}
if pop_district_col == 'DISTRICT_CODE':
    population_data['시군구명'] = population_data[pop_district_col].map(region_mapping)

population_monthly = population_data.groupby(['시군구명', '년월']).agg({
    pop_resident_col: 'sum',
    pop_visit_col: 'sum',
    pop_work_col: 'sum'
}).reset_index()

# 시군구명 열이 없을 경우 생성
if property_district_col != '시군구명':
    property_monthly['시군구명'] = property_monthly[property_district_col]

# 데이터 결합
merged_data = pd.merge(
    property_monthly, 
    population_monthly,
    on=['시군구명', '년월'],
    how='inner'
)

# 새로운 컬럼명으로 변경 (시각화 용이성)
merged_data = merged_data.rename(columns={
    property_price_col: '매매가격(만원/평)',
    property_jeonse_col: '전세가격(만원/평)',
    pop_resident_col: '거주인구',
    pop_visit_col: '방문인구',
    pop_work_col: '근무인구'
})

if merged_data.empty:
    st.warning("지역 및 날짜가 일치하는 데이터가 없습니다. 샘플 데이터를 생성합니다.")
    # 샘플 데이터 생성
    merged_data = pd.DataFrame({
        '시군구명': np.repeat(['서초구', '영등포구', '중구'], 12),
        '년월': np.tile([f'2023{m:02d}' for m in range(1, 13)], 3),
        '매매가격(만원/평)': np.random.normal(3000, 500, 36),
        '전세가격(만원/평)': np.random.normal(2000, 300, 36),
        '거주인구': np.random.normal(50000, 5000, 36),
        '방문인구': np.random.normal(20000, 3000, 36),
        '근무인구': np.random.normal(30000, 4000, 36)
    })

# 데이터 미리보기
st.subheader("결합된 데이터 미리보기")
st.dataframe(merged_data.head())

# 상관관계 분석
st.subheader("상관관계 분석")

# 1. 피어슨 상관계수
st.write("1. 피어슨 상관계수")
correlation_vars = ['매매가격(만원/평)', '전세가격(만원/평)', '거주인구', '방문인구', '근무인구']
correlation_matrix = merged_data[correlation_vars].corr()

# 상관관계 히트맵
fig = px.imshow(
    correlation_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title="주요 변수 간 상관관계 히트맵"
)
st.plotly_chart(fig, use_container_width=True)

# 2. 산점도 행렬
st.write("2. 산점도 행렬")
scatter_matrix_data = merged_data[correlation_vars].copy()

# 데이터 크기를 확인하고 필요시 샘플링
if len(scatter_matrix_data) > 500:
    scatter_matrix_data = scatter_matrix_data.sample(500, random_state=42)

fig = px.scatter_matrix(
    scatter_matrix_data,
    dimensions=correlation_vars,
    color=merged_data['시군구명'].iloc[:len(scatter_matrix_data)],
    title="주요 변수 간 산점도 행렬"
)
st.plotly_chart(fig, use_container_width=True)

# 3. 세부 산점도
st.write("3. 세부 산점도")
x_var = st.selectbox("X축 변수", correlation_vars)
y_var = st.selectbox("Y축 변수", [v for v in correlation_vars if v != x_var])

fig = px.scatter(
    merged_data, 
    x=x_var, 
    y=y_var, 
    color='시군구명',
    trendline='ols',
    title=f"{x_var}와 {y_var}의 관계"
)
st.plotly_chart(fig, use_container_width=True)

# 4. 시간차 상관관계 분석
st.subheader("시간차 상관관계 분석")
st.write("변수 간의 시간차 상관관계를 분석합니다.")

# 시간 정렬
merged_data['년월'] = pd.to_datetime(merged_data['년월'], format='%Y%m')
merged_data = merged_data.sort_values(['시군구명', '년월'])

# 지역 선택
selected_region = st.selectbox(
    "분석할 지역 선택", 
    merged_data['시군구명'].unique()
)

region_data = merged_data[merged_data['시군구명'] == selected_region]

# 시차 설정
max_lag = st.slider("최대 시차(개월)", min_value=1, max_value=12, value=6)

# 시간차 상관관계 계산
lag_results = []
for lag in range(max_lag + 1):
    if lag == 0:
        # 동시 상관관계
        corr_price_pop = region_data['매매가격(만원/평)'].corr(region_data['거주인구'])
        corr_price_visit = region_data['매매가격(만원/평)'].corr(region_data['방문인구'])
        corr_price_work = region_data['매매가격(만원/평)'].corr(region_data['근무인구'])
    else:
        # 시차 상관관계 (가격 → 인구)
        corr_price_pop = region_data['매매가격(만원/평)'].iloc[:-lag].corr(region_data['거주인구'].iloc[lag:])
        corr_price_visit = region_data['매매가격(만원/평)'].iloc[:-lag].corr(region_data['방문인구'].iloc[lag:])
        corr_price_work = region_data['매매가격(만원/평)'].iloc[:-lag].corr(region_data['근무인구'].iloc[lag:])
    
    lag_results.append({
        '시차(개월)': lag,
        '방향': '가격 → 인구',
        '가격→거주인구': corr_price_pop,
        '가격→방문인구': corr_price_visit,
        '가격→근무인구': corr_price_work
    })

    # 인구 → 가격 방향은 시차가 0이 아닐 때만 계산
    if lag > 0:
        corr_pop_price = region_data['거주인구'].iloc[:-lag].corr(region_data['매매가격(만원/평)'].iloc[lag:])
        corr_visit_price = region_data['방문인구'].iloc[:-lag].corr(region_data['매매가격(만원/평)'].iloc[lag:])
        corr_work_price = region_data['근무인구'].iloc[:-lag].corr(region_data['매매가격(만원/평)'].iloc[lag:])
        
        lag_results.append({
            '시차(개월)': lag,
            '방향': '인구 → 가격',
            '거주인구→가격': corr_pop_price,
            '방문인구→가격': corr_visit_price,
            '근무인구→가격': corr_work_price
        })

lag_df = pd.DataFrame(lag_results)

# 시간차 상관관계 시각화
st.write(f"{selected_region}의 시간차 상관관계")
st.dataframe(lag_df)

# 시각화
fig = px.line(
    lag_df, 
    x='시차(개월)', 
    y=['가격→거주인구', '가격→방문인구', '가격→근무인구', '거주인구→가격', '방문인구→가격', '근무인구→가격'],
    color='방향',
    title=f"{selected_region}의 시간차 상관관계",
    labels={'value': '상관계수', 'variable': '관계 방향'}
)
fig.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

# 결과 해석
st.subheader("상관관계 분석 결과 해석")
st.write("""
### 분석 결과 요약
1. **동시 상관관계**: 부동산 가격과 인구 지표 간의 즉각적인 관계를 보여줍니다.
2. **시간차 상관관계**: 한 변수의 변화가 다른 변수에 시간을 두고 영향을 미치는지 보여줍니다.

### 해석 가이드:
- **양의 상관계수**: 한 변수가 증가할 때 다른 변수도 증가함을 의미합니다.
- **음의 상관계수**: 한 변수가 증가할 때 다른 변수는 감소함을 의미합니다.
- **상관계수 크기**: 절대값이 클수록 더 강한 관계를 나타냅니다.
- **시차 피크**: 가장 높은 상관계수를 보이는 시차는 영향이 가장 크게 나타나는 시간을 나타냅니다.

### 인과관계에 대한 단서:
시간차 상관관계 패턴은 인과관계의 방향성에 대한 단서를 제공할 수 있습니다. 예를 들어:
- 부동산 가격 → 인구 방향의 상관계수가 더 높으면, 가격이 인구 분포에 영향을 미칠 가능성이 있습니다.
- 인구 → 부동산 가격 방향의 상관계수가 더 높으면, 인구 변화가 가격에 영향을 미칠 가능성이 있습니다.

그러나 상관관계가 반드시 인과관계를 의미하지는 않으므로, 더 정확한 인과관계 분석을 위해서는 '인과관계 분석' 페이지에서 그랜저 인과관계 검정 등의 방법을 확인하세요.
""")