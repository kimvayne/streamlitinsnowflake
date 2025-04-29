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
        # 컬럼명 한글화
        data = data.rename(columns={
            'BJD_CODE': '법정동코드',
            'EMD': '읍면동명',
            'SD': '시도명',
            'SGG': '시군구명',
            'JEONSE_PRICE_PER_SUPPLY_PYEONG': '전세가격(만원/평)',
            'MEME_PRICE_PER_SUPPLY_PYEONG': '매매가격(만원/평)',
            'YYYYMMDD': '날짜'
        })
        return data
    
    elif dataset == "인구 통계":
        data = get_population_data(region=region)
        # 컬럼명 한글화
        data = data.rename(columns={
            'DISTRICT_CODE': '지역코드',
            'CITY_CODE': '도시코드',
            'AGE_GROUP': '연령대',
            'GENDER': '성별',
            'RESIDENTIAL_POPULATION': '거주인구',
            'VISITING_POPULATION': '방문인구',
            'WORKING_POPULATION': '근무인구',
            'STANDARD_YEAR_MONTH': '기준연월'
        })
        return data
    
    elif dataset == "금융 및 소득":
        data = get_finance_data(region=region)
        # 컬럼명 한글화
        data = data.rename(columns={
            'DISTRICT_CODE': '지역코드',
            'CITY_CODE': '도시코드',
            'AGE_GROUP': '연령대',
            'GENDER': '성별',
            'AVERAGE_INCOME': '평균소득',
            'AVERAGE_HOUSEHOLD_INCOME': '평균가구소득',
            'STANDARD_YEAR_MONTH': '기준연월'
        })
        return data
    
    elif dataset == "백화점 방문":
        data = get_department_store_data(store_name=department_store)
        # 컬럼명 한글화
        data = data.rename(columns={
            'COUNT': '방문객수',
            'DATE_KST': '날짜',
            'DEP_NAME': '백화점명'
        })
        return data
    
    elif dataset == "거주지/직장 위치":
        data = get_home_office_data(department_store=department_store)
        # 컬럼명 한글화
        data = data.rename(columns={
            'ADDR_LV1': '시도',
            'ADDR_LV2': '시군구',
            'ADDR_LV3': '법정동',
            'DEP_NAME': '백화점명',
            'LOC_TYPE': '위치유형',
            'RATIO': '비율'
        })
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
    district_prices = data.groupby('시군구명').agg({
        '매매가격(만원/평)': 'mean',
        '전세가격(만원/평)': 'mean'
    }).reset_index()
    
    fig = px.bar(district_prices, x='시군구명', y=['매매가격(만원/평)', '전세가격(만원/평)'],
                barmode='group', title='시군구별 평균 부동산 가격',
                labels={'value': '가격(만원/평)', 'variable': '가격 유형'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간에 따른 부동산 가격 변화 (시계열)
    st.subheader("시간에 따른 부동산 가격 변화")
    data['날짜'] = pd.to_datetime(data['날짜'])
    time_series = data.groupby([pd.Grouper(key='날짜', freq='M'), '시군구명']).agg({
        '매매가격(만원/평)': 'mean'
    }).reset_index()
    
    fig = px.line(time_series, x='날짜', y='매매가격(만원/평)', color='시군구명',
                 title='시간에 따른 부동산 가격 변화',
                 labels={'매매가격(만원/평)': '평균 매매가격(만원/평)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 히스토그램 - 가격 분포
    st.subheader("부동산 가격 분포")
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        fig = px.histogram(data, x='매매가격(만원/평)', nbins=30, title='매매가격 분포')
        st.plotly_chart(fig, use_container_width=True)
    
    with hist_col2:
        fig = px.histogram(data, x='전세가격(만원/평)', nbins=30, title='전세가격 분포')
        st.plotly_chart(fig, use_container_width=True)

elif dataset == "인구 통계":
    # 지역별 총 인구
    st.subheader("지역별 인구 유형")
    
    # 지역코드와 시군구명 매핑
    region_mapping = {'11650': '서초구', '11560': '영등포구', '11140': '중구'}
    data['시군구명'] = data['지역코드'].map(region_mapping)
    
    # 지역별 인구 집계
    region_pop = data.groupby('시군구명').agg({
        '거주인구': 'sum',
        '방문인구': 'sum',
        '근무인구': 'sum'
    }).reset_index()
    
    fig = px.bar(region_pop, x='시군구명', y=['거주인구', '방문인구', '근무인구'],
                barmode='group', title='지역별 인구 유형',
                labels={'value': '인구 수', 'variable': '인구 유형'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 연령대별 인구 분포
    st.subheader("연령대별 인구 분포")
    age_pop = data.groupby(['시군구명', '연령대']).agg({
        '거주인구': 'sum'
    }).reset_index()
    
    fig = px.bar(age_pop, x='연령대', y='거주인구', color='시군구명',
                barmode='group', title='연령대별 거주인구',
                labels={'거주인구': '인구 수', '연령대': '연령대'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 성별 인구 비율
    st.subheader("성별 인구 비율")
    gender_pop = data.groupby(['시군구명', '성별']).agg({
        '거주인구': 'sum'
    }).reset_index()
    
    fig = px.pie(gender_pop, values='거주인구', names='성별', facet_col='시군구명',
                title='지역별 성별 인구 비율')
    st.plotly_chart(fig, use_container_width=True)

elif dataset == "금융 및 소득":
    # 지역코드와 시군구명 매핑
    region_mapping = {'11650': '서초구', '11560': '영등포구', '11140': '중구'}
    data['시군구명'] = data['지역코드'].map(region_mapping)
    
    # 지역별 평균 소득
    st.subheader("지역별 평균 소득")
    region_income = data.groupby('시군구명').agg({
        '평균소득': 'mean',
        '평균가구소득': 'mean'
    }).reset_index()
    
    fig = px.bar(region_income, x='시군구명', y=['평균소득', '평균가구소득'],
                barmode='group', title='지역별 평균 소득',
                labels={'value': '금액(원)', 'variable': '소득 유형'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 연령대별 평균 소득
    st.subheader("연령대별 평균 소득")
    age_income = data.groupby(['시군구명', '연령대']).agg({
        '평균소득': 'mean'
    }).reset_index()
    
    fig = px.bar(age_income, x='연령대', y='평균소득', color='시군구명',
                barmode='group', title='연령대별 평균 소득',
                labels={'평균소득': '금액(원)', '연령대': '연령대'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 성별 평균 소득
    st.subheader("성별 평균 소득")
    gender_income = data.groupby(['시군구명', '성별']).agg({
        '평균소득': 'mean'
    }).reset_index()
    
    fig = px.bar(gender_income, x='성별', y='평균소득', color='시군구명',
                barmode='group', title='성별 평균 소득',
                labels={'평균소득': '금액(원)', '성별': '성별'})
    st.plotly_chart(fig, use_container_width=True)

elif dataset == "백화점 방문":
    # 백화점별 평균 방문객 수
    st.subheader("백화점별 평균 방문객 수")
    store_visits = data.groupby('백화점명').agg({
        '방문객수': 'mean'
    }).reset_index()
    
    fig = px.bar(store_visits, x='백화점명', y='방문객수',
                title='백화점별 평균 방문객 수',
                labels={'방문객수': '평균 방문객 수', '백화점명': '백화점'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간에 따른 백화점 방문객 수 변화
    st.subheader("시간에 따른 백화점 방문객 수 변화")
    data['날짜'] = pd.to_datetime(data['날짜'])
    time_series = data.groupby([pd.Grouper(key='날짜', freq='D'), '백화점명']).agg({
        '방문객수': 'sum'
    }).reset_index()
    
    fig = px.line(time_series, x='날짜', y='방문객수', color='백화점명',
                 title='시간에 따른 백화점 방문객 수 변화',
                 labels={'방문객수': '방문객 수'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 요일별 방문객 수
    st.subheader("요일별 방문객 수")
    data['요일'] = data['날짜'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_visits = data.groupby(['백화점명', '요일']).agg({
        '방문객수': 'mean'
    }).reset_index()
    
    # 요일 순서 설정
    day_visits['요일_순서'] = day_visits['요일'].apply(lambda x: day_order.index(x) if x in day_order else -1)
    day_visits = day_visits.sort_values('요일_순서')
    
    fig = px.line(day_visits, x='요일', y='방문객수', color='백화점명',
                 title='요일별 평균 방문객 수',
                 labels={'방문객수': '평균 방문객 수', '요일': '요일'})
    st.plotly_chart(fig, use_container_width=True)

elif dataset == "거주지/직장 위치":
    # 위치 유형별 분포
    st.subheader("위치 유형별 분포")
    data['위치유형'] = data['위치유형'].map({1: '거주지', 2: '직장'})
    
    loc_type_ratio = data.groupby(['백화점명', '위치유형']).agg({
        '비율': 'sum'
    }).reset_index()
    
    fig = px.bar(loc_type_ratio, x='백화점명', y='비율', color='위치유형',
                title='백화점별 거주지/직장 위치 비율',
                labels={'비율': '비율', '백화점명': '백화점'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 지역별 거주지/직장 분포
    st.subheader("지역별 거주지/직장 분포")
    region_loc = data.groupby(['시군구', '위치유형']).agg({
        '비율': 'mean'
    }).reset_index()
    
    fig = px.bar(region_loc, x='시군구', y='비율', color='위치유형',
                barmode='group', title='지역별 거주지/직장 분포',
                labels={'비율': '평균 비율', '시군구': '지역'})
    st.plotly_chart(fig, use_container_width=True)
    
    # 백화점별 주요 거주지역
    if '백화점명' in data.columns and data['백화점명'].nunique() > 1:
        st.subheader("백화점별 주요 거주지역")
        
        # 거주지 데이터만 필터링
        home_data = data[data['위치유형'] == '거주지']
        
        # 백화점별 상위 5개 거주지역
        top_regions = home_data.groupby(['백화점명', '시군구'])['비율'].sum().reset_index()
        
        # 각 백화점별로 상위 5개 지역 선택
        top5_regions = []
        for store in top_regions['백화점명'].unique():
            store_data = top_regions[top_regions['백화점명'] == store]
            top5 = store_data.nlargest(5, '비율')
            top5_regions.append(top5)
        
        if top5_regions:
            top5_df = pd.concat(top5_regions)
            
            fig = px.bar(top5_df, x='시군구', y='비율', color='백화점명', facet_col='백화점명',
                        title='백화점별 주요 거주지역(상위 5개)',
                        labels={'비율': '비율', '시군구': '지역'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

# 데이터 요약 통계
st.subheader("데이터 요약 통계")
if dataset in ["부동산 가격", "인구 통계", "금융 및 소득", "백화점 방문"]:
    # 수치형 열만 선택
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        st.dataframe(data[numeric_cols].describe())
    else:
        st.info("수치형 데이터가 없습니다.")
else:
    st.info("이 데이터셋은 요약 통계를 제공하지 않습니다.")

# 상관관계 분석 (수치형 데이터가 충분한 경우)
if dataset in ["부동산 가격", "인구 통계", "금융 및 소득"]:
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("상관관계 분석")
        correlation = data[numeric_cols].corr()
        
        # Plotly로 히트맵 구현
        fig = px.imshow(correlation, 
                      text_auto=True, 
                      color_continuous_scale='RdBu_r',
                      title='변수 간 상관관계')
        st.plotly_chart(fig, use_container_width=True)

# 데이터셋 통합 분석 (추가 인사이트)
if st.checkbox("통합 분석 보기"):
    st.subheader("데이터셋 통합 분석")
    
    try:
        # 부동산 가격 데이터 로드
        property_data = get_property_data(region=region)
        property_data = property_data.rename(columns={
            'BJD_CODE': '법정동코드',
            'EMD': '읍면동명',
            'SD': '시도명',
            'SGG': '시군구명',
            'JEONSE_PRICE_PER_SUPPLY_PYEONG': '전세가격(만원/평)',
            'MEME_PRICE_PER_SUPPLY_PYEONG': '매매가격(만원/평)',
            'YYYYMMDD': '날짜'
        })
        
        # 인구 통계 데이터 로드
        population_data = get_population_data(region=region)
        population_data = population_data.rename(columns={
            'DISTRICT_CODE': '지역코드',
            'CITY_CODE': '도시코드',
            'AGE_GROUP': '연령대',
            'GENDER': '성별',
            'RESIDENTIAL_POPULATION': '거주인구',
            'VISITING_POPULATION': '방문인구',
            'WORKING_POPULATION': '근무인구',
            'STANDARD_YEAR_MONTH': '기준연월'
        })
        
        # 지역코드와 시군구명 매핑
        region_mapping = {'11650': '서초구', '11560': '영등포구', '11140': '중구'}
        population_data['시군구명'] = population_data['지역코드'].map(region_mapping)
        
        # 월별 데이터 집계
        property_data['년월'] = pd.to_datetime(property_data['날짜']).dt.strftime('%Y%m')
        property_monthly = property_data.groupby(['시군구명', '년월']).agg({
            '매매가격(만원/평)': 'mean'
        }).reset_index()
        
        population_monthly = population_data.groupby(['시군구명', '기준연월']).agg({
            '거주인구': 'sum',
            '방문인구': 'sum'
        }).reset_index()
        population_monthly = population_monthly.rename(columns={'기준연월': '년월'})
        
        # 데이터 병합
        merged_data = pd.merge(
            property_monthly, 
            population_monthly,
            on=['시군구명', '년월'],
            how='inner'
        )
        
        if not merged_data.empty:
            st.write("부동산 가격과 인구 데이터 통합 분석")
            
            # 산점도 - 부동산 가격 vs 거주인구
            st.subheader("부동산 가격과 거주인구의 관계")
            fig = px.scatter(merged_data, x='거주인구', y='매매가격(만원/평)', 
                           color='시군구명', size='방문인구',
                           title='부동산 가격과 거주인구의 관계',
                           labels={'거주인구': '거주인구', '매매가격(만원/평)': '매매가격(만원/평)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 시계열 분석 - 시간에 따른 변화
            st.subheader("시간에 따른 부동산 가격과 인구 변화")
            merged_data['년월'] = pd.to_datetime(merged_data['년월'], format='%Y%m')
            
            # 시군구별 분석
            for district in merged_data['시군구명'].unique():
                district_data = merged_data[merged_data['시군구명'] == district]
                
                # 이중 Y축 그래프
                fig = go.Figure()
                
                # 부동산 가격
                fig.add_trace(go.Scatter(
                    x=district_data['년월'], 
                    y=district_data['매매가격(만원/평)'],
                    name='매매가격(만원/평)',
                    line=dict(color='blue')
                ))
                
                # 인구수 (보조 y축)
                fig.add_trace(go.Scatter(
                    x=district_data['년월'], 
                    y=district_data['거주인구'],
                    name='거주인구',
                    line=dict(color='red'),
                    yaxis='y2'
                ))
                
                # 레이아웃 설정
                fig.update_layout(
                    title=f'{district} - 부동산 가격과 인구 변화',
                    xaxis=dict(title='날짜'),
                    yaxis=dict(
                        title='매매가격(만원/평)',
                        titlefont=dict(color='blue'),
                        tickfont=dict(color='blue')
                    ),
                    yaxis2=dict(
                        title='거주인구',
                        titlefont=dict(color='red'),
                        tickfont=dict(color='red'),
                        anchor='x',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(x=0.01, y=0.99),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("통합 분석을 위한 데이터가 충분하지 않습니다.")
    
    except Exception as e:
        st.error(f"통합 분석 중 오류가 발생했습니다: {e}")
        st.info("기본 데이터 분석 결과만 표시합니다.")

# 분석 결과 다운로드
st.subheader("데이터 다운로드")
csv = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="CSV로 다운로드",
    data=csv,
    file_name=f"{dataset}_data.csv",
    mime="text/csv",
)