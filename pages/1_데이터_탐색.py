import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import (
    get_property_data, 
    get_population_data, 
    get_finance_data, 
    get_department_store_data,
    get_home_office_data
)

st.set_page_config(page_title="데이터 탐색", page_icon="📊")

st.title("데이터 탐색")
st.write("토지가격과 인구밀집도 분석을 위한 데이터를 탐색합니다.")

# 사이드바 - 데이터셋 선택
st.sidebar.header("데이터셋 선택")
dataset = st.sidebar.selectbox(
    "탐색할 데이터셋",
    ["부동산 가격", "인구 통계", "금융 및 소득", "백화점 방문", "거주지/직장 위치"]
)

# 사이드바 - 지역 선택
region = None
department_store = None
if dataset in ["부동산 가격", "인구 통계", "금융 및 소득"]:
    region = st.sidebar.selectbox(
        "지역 선택",
        ["전체", "서초구", "영등포구", "중구"]
    )
    if region == "전체":
        region = None
elif dataset in ["백화점 방문", "거주지/직장 위치"]:
    department_store = st.sidebar.selectbox(
        "백화점 선택",
        ["전체", "롯데백화점", "현대백화점", "신세계백화점"]
    )
    if department_store == "전체":
        department_store = None

# 데이터셋 로드
@st.cache_data
def load_selected_dataset():
    if dataset == "부동산 가격":
        data = get_property_data(region=region)
        return data
    
    elif dataset == "인구 통계":
        data = get_population_data(region=region)
        return data
    
    elif dataset == "금융 및 소득":
        data = get_finance_data(region=region)
        return data
    
    elif dataset == "백화점 방문":
        data = get_department_store_data(store_name=department_store)
        return data
    
    elif dataset == "거주지/직장 위치":
        data = get_home_office_data(department_store=department_store)
        return data

data = load_selected_dataset()

# 데이터 기본 정보 표시
st.subheader("데이터 미리보기")
st.dataframe(data.head(10))

st.subheader("데이터 기본 정보")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**행 수:** {data.shape[0]:,}")
    st.write(f"**열 수:** {data.shape[1]}")
with col2:
    st.write(f"**결측치 수:** {data.isna().sum().sum():,}")
    st.write(f"**중복 행 수:** {data.duplicated().sum():,}")

# 데이터 유형별 시각화 및 분석
st.subheader("데이터 탐색 및 시각화")

if dataset == "부동산 가격":
    # 시군구별 평균 부동산 가격
    st.subheader("시군구별 평균 부동산 가격")
    
    # 지역명 컬럼 확인 및 필요시 매핑
    if 'SGG' in data.columns:
        district_col = 'SGG'
    else:
        district_col = '시군구명'
        
    # 가격 컬럼 확인
    if 'MEME_PRICE_PER_SUPPLY_PYEONG' in data.columns:
        price_col = 'MEME_PRICE_PER_SUPPLY_PYEONG'
        jeonse_col = 'JEONSE_PRICE_PER_SUPPLY_PYEONG'
    else:
        price_col = '매매가격(만원/평)'
        jeonse_col = '전세가격(만원/평)'
    
    # 날짜 컬럼 확인
    if 'YYYYMMDD' in data.columns:
        date_col = 'YYYYMMDD'
    else:
        date_col = '날짜'
    
    # 데이터 집계
    district_prices = data.groupby(district_col).agg({
        price_col: 'mean',
        jeonse_col: 'mean'
    }).reset_index()
    
    # 컬럼명 한글화 (시각화용)
    district_prices = district_prices.rename(columns={
        district_col: '지역',
        price_col: '매매가격(만원/평)',
        jeonse_col: '전세가격(만원/평)'
    })
    
    fig = px.bar(district_prices, x='지역', y=['매매가격(만원/평)', '전세가격(만원/평)'],
                barmode='group', title='시군구별 평균 부동산 가격',
                labels={'value': '가격(만원/평)', 'variable': '가격 유형'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간에 따른 부동산 가격 변화 (시계열)
    st.subheader("시간에 따른 부동산 가격 변화")
    
    # 날짜 변환
    data[date_col] = pd.to_datetime(data[date_col])
    
    # 월별 평균 집계
    time_series = data.groupby([pd.Grouper(key=date_col, freq='M'), district_col]).agg({
        price_col: 'mean'
    }).reset_index()
    
    # 컬럼명 한글화 (시각화용)
    time_series = time_series.rename(columns={
        district_col: '지역',
        price_col: '평균 매매가격(만원/평)',
        date_col: '날짜'
    })
    
    fig = px.line(time_series, x='날짜', y='평균 매매가격(만원/평)', color='지역',
                 title='시간에 따른 부동산 가격 변화',
                 labels={'평균 매매가격(만원/평)': '평균 매매가격(만원/평)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 히스토그램 - 가격 분포
    st.subheader("부동산 가격 분포")
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        fig = px.histogram(data, x=price_col, nbins=30, title='매매가격 분포')
        fig.update_layout(xaxis_title='매매가격(만원/평)')
        st.plotly_chart(fig, use_container_width=True)
    
    with hist_col2:
        fig = px.histogram(data, x=jeonse_col, nbins=30, title='전세가격 분포')
        fig.update_layout(xaxis_title='전세가격(만원/평)')
        st.plotly_chart(fig, use_container_width=True)

elif dataset == "인구 통계":
    # 지역별 총 인구
    st.subheader("지역별 인구 유형")
    
    # 컬럼명 확인
    if 'DISTRICT_CODE' in data.columns:
        district_col = 'DISTRICT_CODE'
        pop_cols = ['RESIDENTIAL_POPULATION', 'VISITING_POPULATION', 'WORKING_POPULATION']
    else:
        district_col = '지역코드'
        pop_cols = ['거주인구', '방문인구', '근무인구']
    
    # 지역코드와 시군구명 매핑
    region_mapping = {'11650': '서초구', '11560': '영등포구', '11140': '중구'}
    data['시군구명'] = data[district_col].map(region_mapping)
    
    # 지역별 인구 집계
    region_pop = data.groupby('시군구명').agg({
        pop_cols[0]: 'sum',
        pop_cols[1]: 'sum',
        pop_cols[2]: 'sum'
    }).reset_index()
    
    # 컬럼명 한글화 (시각화용)
    region_pop = region_pop.rename(columns={
        pop_cols[0]: '거주인구',
        pop_cols[1]: '방문인구',
        pop_cols[2]: '근무인구'
    })
    
    fig = px.bar(region_pop, x='시군구명', y=['거주인구', '방문인구', '근무인구'],
                barmode='group', title='지역별 인구 유형',
                labels={'value': '인구 수', 'variable': '인구 유형'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 연령대별 인구 분포
    st.subheader("연령대별 인구 분포")
    
    # 연령대 컬럼 확인
    if 'AGE_GROUP' in data.columns:
        age_col = 'AGE_GROUP'
    else:
        age_col = '연령대'
        
    # 연령대별 인구 집계
    age_pop = data.groupby(['시군구명', age_col]).agg({
        pop_cols[0]: 'sum'
    }).reset_index()
    
    # 컬럼명 한글화 (시각화용)
    age_pop = age_pop.rename(columns={
        age_col: '연령대',
        pop_cols[0]: '거주인구'
    })
    
    fig = px.bar(age_pop, x='연령대', y='거주인구', color='시군구명',
                barmode='group', title='연령대별 거주인구',
                labels={'거주인구': '인구 수', '연령대': '연령대'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 성별 인구 비율
    st.subheader("성별 인구 비율")
    
    # 성별 컬럼 확인
    if 'GENDER' in data.columns:
        gender_col = 'GENDER'
    else:
        gender_col = '성별'
        
    # 성별 인구 집계
    gender_pop = data.groupby(['시군구명', gender_col]).agg({
        pop_cols[0]: 'sum'})