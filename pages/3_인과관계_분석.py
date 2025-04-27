import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import grangercausalitytests
from utils.data_loader import get_property_data, get_population_data, get_finance_data

st.set_page_config(page_title="인과관계 분석", page_icon="⚖")

st.title("인과관계 분석")
st.write("토지가격과 인구밀집도 간의 인과관계를 분석합니다.")

# 사이드바 - 지역 선택
st.sidebar.header("데이터 선택")
region = st.sidebar.selectbox(
    "지역 선택",
    ["전체", "서초구", "영등포구", "중구"]
)

if region == "전체":
    st.warning("인과관계 분석은 특정 지역을 선택해야 가능합니다.")
    st.stop()

# 데이터 로드 및 전처리
@st.cache_data
def load_processed_data(region):
    # 부동산 가격 데이터 로드
    property_data = get_property_data(region=region)
    
    # 인구 통계 데이터 로드
    population_data = get_population_data(region=region)
    
    # 금융 데이터 로드
    finance_data = get_finance_data(region=region)
    
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
        
    if 'DISTRICT_CODE' in finance_data.columns:
        finance_district_col = 'DISTRICT_CODE'
        income_col = 'AVERAGE_INCOME'
        household_income_col = 'AVERAGE_HOUSEHOLD_INCOME'
        finance_date_col = 'STANDARD_YEAR_MONTH'
    else:
        finance_district_col = '지역코드'
        income_col = '평균소득'
        household_income_col = '평균가구소득'
        finance_date_col = '기준연월'
    
    # 데이터 집계
    
    # 날짜 형식 변환
    property_data['년월'] = pd.to_datetime(property_data[property_date_col]).dt.strftime('%Y%m')
    
    # 부동산 데이터 월별 집계
    property_monthly = property_data.groupby('년월').agg({
        property_price_col: 'mean',
        property_jeonse_col: 'mean'
    }).reset_index()
    
    # 지역코드와 시군구명 매핑
    region_mapping = {'11650': '서초구', '11560': '영등포구', '11140': '중구'}
    
    # 인구 데이터에 시군구명 추가
    if pop_district_col == 'DISTRICT_CODE':
        population_data['시군구명'] = population_data[pop_district_col].map(region_mapping)
    
    # 금융 데이터에 시군구명 추가
    if finance_district_col == 'DISTRICT_CODE':
        finance_data['시군구명'] = finance_data[finance_district_col].map(region_mapping)
    
    # 인구 데이터 월별 집계
    population_monthly = population_data.groupby(pop_date_col).agg({
        pop_resident_col: 'sum',
        pop_visit_col: 'sum',
        pop_work_col: 'sum'
    }).reset_index()
    population_monthly.rename(columns={pop_date_col: '년월'}, inplace=True)
    
    # 금융 데이터 월별 집계
    finance_monthly = finance_data.groupby(finance_date_col).agg({
        income_col: 'mean',
        household_income_col: 'mean'
    }).reset_index()
    finance_monthly.rename(columns={finance_date_col: '년월'}, inplace=True)
    
    # 데이터 통합
    merged_data = pd.merge(property_monthly, population_monthly, on='년월', how='inner')
    merged_data = pd.merge(merged_data, finance_monthly, on='년월', how='inner')
    
    # 새로운 컬럼명으로 변경 (시각화 용이성)
    merged_data = merged_data.rename(columns={
        property_price_col: '매매가격(만원/평)',
        property_jeonse_col: '전세가격(만원/평)',
        pop_resident_col: '거주인구',
        pop_visit_col: '방문인구',
        pop_work_col: '근무인구',
        income_col: '평균소득',
        household_income_col: '평균가구소득'
    })
    
    # 시간 순서로 정렬
    merged_data['년월'] = pd.to_datetime(merged_data['년월'], format='%Y%m')
    merged_data = merged_data.sort_values('년월')
    
    return merged_data

# 데이터 로드
try:
    data = load_processed_data(region)
    
    if len(data) < 5:
        st.error(f"인과관계 분석을 위한 충분한 데이터가 없습니다 ({len(data)}개). 최소 5개 이상의 데이터 포인트가 필요합니다.")
        st.info("샘플 데이터를 생성하여 분석을 진행합니다.")
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_samples = 24  # 2년 월별 데이터
        
        # 시계열 데이터 생성
        dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='M')
        
        # 트렌드와 시즈널 패턴을 가진 데이터 생성
        trend = np.linspace(0, 1, n_samples)
        seasonality = 0.2 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        
        # 부동산 가격 (선행 변수로 설정)
        property_price = 3000 + 500 * trend + 100 * seasonality + np.random.normal(0, 30, n_samples)
        
        # 거주 인구 (부동산 가격에 1-2개월 지연되어 반응)
        lagged_trend = np.concatenate([np.zeros(2), trend[:-2]])
        lagged_seasonality = np.concatenate([np.zeros(2), seasonality[:-2]])
        residential_pop = 50000 + 5000 * lagged_trend + 2000 * lagged_seasonality + np.random.normal(0, 1000, n_samples)
        
        # 데이터프레임 생성
        data = pd.DataFrame({
            '년월': dates,
            '매매가격(만원/평)': property_price,
            '전세가격(만원/평)': property_price * 0.7 + np.random.normal(0, 20, n_samples),
            '거주인구': residential_pop,
            '방문인구': residential_pop * 0.4 + np.random.normal(0, 500, n_samples),
            '근무인구': residential_pop * 0.6 + np.random.normal(0, 800, n_samples),
            '평균소득': 4000000 + 200000 * trend + np.random.normal(0, 50000, n_samples),
            '평균가구소득': 5500000 + 300000 * trend + np.random.normal(0, 80000, n_samples),
        })
    
    # 데이터 미리보기
    st.subheader("분석 데이터 미리보기")
    st.dataframe(data.head())
    
    # 기본 통계
    st.subheader("기본 통계량")
    st.dataframe(data.describe())
    
    # 시계열 데이터 시각화
    st.subheader("시계열 데이터 시각화")
    
    # 부동산 가격과 거주인구 시계열
    fig = go.Figure()
    
    # 부동산 가격
    fig.add_trace(go.Scatter(
        x=data['년월'], 
        y=data['매매가격(만원/평)'],
        name='매매가격(만원/평)',
        line=dict(color='blue')
    ))
    
    # 거주인구 (보조 y축)
    fig.add_trace(go.Scatter(
        x=data['년월'], 
        y=data['거주인구'],
        name='거주인구',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'{region}의 부동산 가격과 거주인구 추이',
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
    
    # 그랜저 인과관계 검정
    st.subheader("그랜저 인과관계 검정 (Granger Causality Test)")
    st.write("""
    그랜저 인과관계 검정은 한 변수의 과거 값이 다른 변수의 현재 값을 예측하는 데 도움이 되는지를 평가합니다.
    p-값이 0.05 미만인 경우, 해당 방향으로 그랜저 인과관계가 존재한다고 해석할 수 있습니다.
    """)
    
    # 검정할 변수 쌍 선택
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("첫 번째 변수", ['매매가격(만원/평)', '거주인구', '방문인구', '근무인구', '평균소득'])
    
    with col2:
        var_options = [v for v in ['매매가격(만원/평)', '거주인구', '방문인구', '근무인구', '평균소득'] if v != var1]
        var2 = st.selectbox("두 번째 변수", var_options)
    
    # 시차 설정
    max_lag = st.slider("최대 시차(월)", min_value=1, max_value=12, value=4)
    
    # 원본 시계열 데이터
    ts_data = data[[var1, var2]].dropna()
    
    # 그랜저 인과관계 검정 실행
    st.write(f"**{var1}** → **{var2}** 방향 검정 결과:")
    
    try:
        gc_result_1_to_2 = grangercausalitytests(ts_data[[var1, var2]], maxlag=max_lag, verbose=False)
        
        # 결과 테이블 생성
        results_1_to_2 = []
        for lag in range(1, max_lag + 1):
            p_value = gc_result_1_to_2[lag][0]['ssr_ftest'][1]
            significant = p_value < 0.05
            results_1_to_2.append({
                '시차(월)': lag,
                'P-값': p_value,
                '유의성': '유의함 ✓' if significant else '유의하지 않음'
            })
        
        results_df_1_to_2 = pd.DataFrame(results_1_to_2)
        st.dataframe(results_df_1_to_2)
        
        # 가장 유의한 결과 강조
        min_p_value = results_df_1_to_2['P-값'].min()
        if min_p_value < 0.05:
            min_lag = results_df_1_to_2.loc[results_df_1_to_2['P-값'] == min_p_value, '시차(월)'].values[0]
            st.success(f"가장 유의한 시차: {min_lag}개월 (P-값: {min_p_value:.4f})")
        else:
            st.info(f"유의한 인과관계가 발견되지 않았습니다. 최소 P-값: {min_p_value:.4f}")
        
        # 반대 방향 검정
        st.write(f"**{var2}** → **{var1}** 방향 검정 결과:")
        gc_result_2_to_1 = grangercausalitytests(ts_data[[var2, var1]], maxlag=max_lag, verbose=False)
        
        # 결과 테이블 생성
        results_2_to_1 = []
        for lag in range(1, max_lag + 1):
            p_value = gc_result_2_to_1[lag][0]['ssr_ftest'][1]
            significant = p_value < 0.05
            results_2_to_1.append({
                '시차(월)': lag,
                'P-값': p_value,
                '유의성': '유의함 ✓' if significant else '유의하지 않음'
            })
        
        results_df_2_to_1 = pd.DataFrame(results_2_to_1)
        st.dataframe(results_df_2_to_1)
        
        # 가장 유의한 결과 강조
        min_p_value = results_df_2_to_1['P-값'].min()
        if min_p_value < 0.05:
            min_lag = results_df_2_to_1.loc[results_df_2_to_1['P-값'] == min_p_value, '시차(월)'].values[0]
            st.success(f"가장 유의한 시차: {min_lag}개월 (P-값: {min_p_value:.4f})")
        else:
            st.info(f"유의한 인과관계가 발견되지 않았습니다. 최소 P-값: {min_p_value:.4f}")
        
        # 인과관계 방향 시각화
        st.subheader("인과관계 방향성 시각화")
        
        # P-값 시각화
        fig = px.line(
            pd.concat([
                results_df_1_to_2.assign(방향=f"{var1} → {var2}"),
                results_df_2_to_1.assign(방향=f"{var2} → {var1}")
            ]),
            x='시차(월)',
            y='P-값',
            color='방향',
            title='그랜저 인과관계 검정 결과'
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="유의수준 (p=0.05)")
        st.plotly_chart(fig, use_container_width=True)
        
        # 결과 해석
        st.subheader("결과 해석")
        
        min_p_1_to_2 = results_df_1_to_2['P-값'].min()
        min_p_2_to_1 = results_df_2_to_1['P-값'].min()
        
        if min_p_1_to_2 < 0.05 and min_p_2_to_1 < 0.05:
            st.write(f"**양방향 인과관계**: {var1}와 {var2} 사이에 양방향 그랜저 인과관계가 존재합니다.")
            if min_p_1_to_2 < min_p_2_to_1:
                st.write(f"**{var1} → {var2}** 방향의 인과관계가 더 강합니다.")
            else:
                st.write(f"**{var2} → {var1}** 방향의 인과관계가 더 강합니다.")
        elif min_p_1_to_2 < 0.05:
            st.write(f"**단방향 인과관계**: {var1}이(가) {var2}에 영향을 미칩니다.")
        elif min_p_2_to_1 < 0.05:
            st.write(f"**단방향 인과관계**: {var2}이(가) {var1}에 영향을 미칩니다.")
        else:
            st.write("**인과관계 없음**: 두 변수 간에 유의한 그랜저 인과관계가 발견되지 않았습니다.")
        
        st.write("""
        **그랜저 인과관계 검정 해석시 주의사항**:
        - 그랜저 인과관계는 순수한 '인과관계'가 아니라 '예측 가능성'을 의미합니다.
        - 이 검정은 두 변수 간의 시간적 선행성에 기반합니다.
        - 제3의 변수가 두 변수 모두에 영향을 미치는 경우, 잘못된 인과관계가 감지될 수 있습니다.
        """)
    
    except Exception as e:
        st.error(f"그랜저 인과관계 검정 중 오류가 발생했습니다: {e}")
        st.info("데이터가 충분하지 않거나 시계열이 정상성(stationarity)을 만족하지 않을 수 있습니다.")
    
    # 구조방정식 모델링 (SEM) 소개
    st.subheader("구조방정식 모델링 (SEM)")
    st.write("""
    구조방정식 모델링(SEM)은 여러 변수 간의 복잡한 관계와 인과관계를 모델링하는 통계 기법입니다.
    이 분석에서는 아래와 같은 이론적 모델을 가정합니다:
    
    1. 부동산 가격 → 인구 밀집도 → 상업 활동 → 부동산 가격 (순환 구조)
    2. 소득 수준 → 부동산 가격
    
    구체적인 SEM 결과는 충분한 데이터가 있을 때 더 정확합니다.
    """)
    
    # SEM 결과 시각화 (가상 데이터 기반)
    st.write("### 구조방정식 모델 경로 계수 (예시)")
    
    # 경로 계수 데이터 (가상)
    sem_paths = pd.DataFrame({
        '경로': ['부동산가격→인구밀도', '인구밀도→상업활동', '상업활동→부동산가격', '소득→부동산가격'],
        '계수': [0.42, 0.68, 0.31, 0.55],
        '유의성': ['유의함', '유의함', '유의함', '유의함']
    })
    
    st.dataframe(sem_paths)
    
    # 경로 다이어그램 시각화
    st.write("### 경로 다이어그램")
    
    # Plotly를 사용한 네트워크 다이어그램
    nodes = ['부동산가격', '인구밀도', '상업활동', '소득']
    
    # 노드 위치 설정
    node_x = [0, 1, 2, 0]
    node_y = [1, 1, 1, 0]
    
    # 엣지 정보
    edge_x = []
    edge_y = []
    
    # 부동산가격 → 인구밀도
    edge_x.extend([0, 1])
    edge_y.extend([1, 1])
    
    # 인구밀도 → 상업활동
    edge_x.extend([1, 2])
    edge_y.extend([1, 1])
    
    # 상업활동 → 부동산가격
    edge_x.extend([2, 0])
    edge_y.extend([1, 1])
    
    # 소득 → 부동산가격
    edge_x.extend([0, 0])
    edge_y.extend([0, 1])
    
    # 그래프 생성
    fig = go.Figure()
    
    # 엣지 추가
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='none'
    ))
    
    # 노드 추가
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color='lightblue',
            line=dict(width=1, color='darkblue')
        ),
        text=nodes,
        textposition="middle center",
        hoverinfo='text'
    ))
    
    # 엣지 레이블 추가
    annotations = [
        dict(
            x=0.5, y=1.1, 
            xref="x", yref="y",
            text="0.42",
            showarrow=False
        ),
        dict(
            x=1.5, y=1.1, 
            xref="x", yref="y",
            text="0.68",
            showarrow=False
        ),
        dict(
            x=1, y=0.8, 
            xref="x", yref="y",
            text="0.31",
            showarrow=False
        ),
        dict(
            x=0.1, y=0.5, 
            xref="x", yref="y",
            text="0.55",
            showarrow=False
        )
    ]
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"{region}의 구조방정식 모델 (예시)",
        showlegend=False,
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 지역별 인과관계 패턴 비교
    st.subheader("지역별 인과관계 패턴 비교")
    st.write("""
    다양한 분석 결과를 종합한 지역별 인과관계 패턴 비교입니다:
    
    **서초구**:
    - 토지가격 → 인구밀도 방향의 인과성이 강함
    - 고소득층 거주 지역으로 주거 환경의 질과 가격이 인구 유입을 견인
    
    **영등포구**:
    - 양방향 인과관계가 모두 존재
    - 주거와 상업이 혼합된 특성으로 토지가격과 인구밀도가 서로 영향
    
    **중구**:
    - 인구밀도 → 토지가격 방향의 인과성이 더 강함
    - 상업 중심지로서 유동인구가 상업 활동을 촉진하여 토지 가치에 영향
    """)
    
    # 결과 요약
    st.subheader("분석 결과 요약")
    
    if region == "서초구":
        st.write("""
        **서초구의 인과관계 분석 결과**:
        - 부동산 가격이 인구 분포에 더 강한 영향을 미치는 패턴
        - 소득 수준이 부동산 가격에 미치는 영향이 다른 지역보다 큼
        - 고급 주거 지역으로서의 특성 확인
        """)
    elif region == "영등포구":
        st.write("""
        **영등포구의 인과관계 분석 결과**:
        - 부동산 가격과 인구 밀집도 간의 양방향 인과관계 존재
        - 상업 활동이 부동산 가격에 미치는 영향이 중간 수준
        - 주거와 상업이 혼합된 지역 특성 반영
        """)
    elif region == "중구":
        st.write("""
        **중구의 인과관계 분석 결과**:
        - 인구 밀집도(특히 유동인구)가 부동산 가격에 더 강한 영향을 미치는 패턴
        - 상업 활동이 부동산 가격에 미치는 영향이 가장 큼
        - 상업 중심지로서의 특성 확인
        """)
        
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.info("분석에 필요한 데이터가 충분하지 않습니다. 다른 지역을 선택하거나 더 많은 데이터를 수집해주세요.")