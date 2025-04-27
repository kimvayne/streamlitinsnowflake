import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import get_property_data, get_population_data, get_finance_data

st.set_page_config(page_title="예측 모델", page_icon="🔮")

st.title("예측 모델")
st.write("토지가격과 인구밀집도 데이터를 기반으로 예측 모델을 개발합니다.")

# 사이드바 - 옵션 선택
st.sidebar.header("모델링 옵션")
target_variable = st.sidebar.selectbox(
    "예측 대상 변수",
    ["부동산 가격", "인구 변화"]
)

region = st.sidebar.selectbox(
    "지역 선택",
    ["서초구", "영등포구", "중구"]
)

# 데이터 로드 및 전처리
@st.cache_data
def load_processed_data(region):
    try:
        # 부동산 가격 데이터 로드
        property_data = get_property_data(region=region)
        property_data.rename(columns={
            'BJD_CODE': '법정동코드',
            'EMD': '읍면동명',
            'SD': '시도명',
            'SGG': '시군구명',
            'JEONSE_PRICE_PER_SUPPLY_PYEONG': '전세가격(만원/평)',
            'MEME_PRICE_PER_SUPPLY_PYEONG': '매매가격(만원/평)',
            'YYYYMMDD': '날짜'
        }, inplace=True)
        
        # 인구 통계 데이터 로드
        population_data = get_population_data(region=region)
        population_data.rename(columns={
            'DISTRICT_CODE': '지역코드',
            'CITY_CODE': '도시코드',
            'AGE_GROUP': '연령대',
            'GENDER': '성별',
            'RESIDENTIAL_POPULATION': '거주인구',
            'VISITING_POPULATION': '방문인구',
            'WORKING_POPULATION': '근무인구',
            'STANDARD_YEAR_MONTH': '기준연월'
        }, inplace=True)
        
        # 금융 데이터 로드
        finance_data = get_finance_data(region=region)
        finance_data.rename(columns={
            'DISTRICT_CODE': '지역코드',
            'CITY_CODE': '도시코드',
            'AGE_GROUP': '연령대',
            'GENDER': '성별',
            'AVERAGE_INCOME': '평균소득',
            'AVERAGE_HOUSEHOLD_INCOME': '평균가구소득',
            'STANDARD_YEAR_MONTH': '기준연월'
        }, inplace=True)
        
        # 지역코드-시군구명 매핑
        region_mapping = {'11650': '서초구', '11560': '영등포구', '11140': '중구'}
        population_data['시군구명'] = population_data['지역코드'].map(region_mapping)
        finance_data['시군구명'] = finance_data['지역코드'].map(region_mapping)
        
        # 날짜 형식 변환
        property_data['년월'] = pd.to_datetime(property_data['날짜']).dt.strftime('%Y%m')
        
        # 부동산 데이터 월별 집계
        property_monthly = property_data.groupby('년월').agg({
            '매매가격(만원/평)': 'mean',
            '전세가격(만원/평)': 'mean'
        }).reset_index()
        
        # 인구 데이터 월별 집계
        population_monthly = population_data.groupby('기준연월').agg({
            '거주인구': 'sum',
            '방문인구': 'sum',
            '근무인구': 'sum'
        }).reset_index()
        population_monthly.rename(columns={'기준연월': '년월'}, inplace=True)
        
        # 금융 데이터 월별 집계
        finance_monthly = finance_data.groupby('기준연월').agg({
            '평균소득': 'mean',
            '평균가구소득': 'mean'
        }).reset_index()
        finance_monthly.rename(columns={'기준연월': '년월'}, inplace=True)
        
        # 데이터 통합 - 두 단계로 병합
        merged_data1 = pd.merge(
            property_monthly, 
            population_monthly,
            on=['년월'],
            how='inner'
        )
        
        merged_data = pd.merge(
            merged_data1,
            finance_monthly,
            on=['년월'],
            how='inner'
        )
        
        # 시간 순서로 정렬
        merged_data['년월'] = pd.to_datetime(merged_data['년월'], format='%Y%m')
        merged_data = merged_data.sort_values('년월')
        
        return merged_data
    
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# 데이터 준비
try:
    data = load_processed_data(region)
    
    if data.empty or len(data) < 10:
        st.warning("충분한 데이터가 없어 샘플 데이터를 생성합니다.")
        
        # 샘플 데이터 생성
        np.random.seed(42)
        n_samples = 36  # 3년 월별 데이터
        
        # 약간의 추세와 계절성을 가진 시계열 생성
        trend = np.linspace(0, 1, n_samples)
        seasonality = 0.1 * np.sin(np.linspace(0, 6*np.pi, n_samples))
        
        data = pd.DataFrame({
            '년월': pd.date_range(start='2021-01-01', periods=n_samples, freq='M'),
            '매매가격(만원/평)': 3000 + 500 * trend + 100 * seasonality + np.random.normal(0, 50, n_samples),
            '전세가격(만원/평)': 2000 + 300 * trend + 80 * seasonality + np.random.normal(0, 30, n_samples),
            '거주인구': 50000 + 3000 * trend + 1000 * seasonality + np.random.normal(0, 500, n_samples),
            '방문인구': 20000 + 2000 * trend + 1500 * seasonality + np.random.normal(0, 400, n_samples),
            '근무인구': 30000 + 2500 * trend + 1200 * seasonality + np.random.normal(0, 450, n_samples),
            '평균소득': 4500000 + 300000 * trend + np.random.normal(0, 100000, n_samples),
            '평균가구소득': 6500000 + 400000 * trend + np.random.normal(0, 150000, n_samples)
        })
    
    # 시간 관련 특성 추가
    data['month'] = data['년월'].dt.month
    data['year'] = data['년월'].dt.year
    
    # 데이터 미리보기
    st.subheader("데이터 미리보기")
    st.dataframe(data.head())
    
    # 기본 통계
    st.subheader("기본 통계량")
    st.dataframe(data.describe())
    
    # 시계열 데이터 시각화
    st.subheader("시계열 데이터 시각화")
    
    # 부동산 가격 시각화
    price_chart = pd.DataFrame({
        '날짜': data['년월'],
        '매매가격(만원/평)': data['매매가격(만원/평)']
    })
    st.line_chart(price_chart.set_index('날짜'))
    
    # 인구 데이터 시각화
    pop_chart = pd.DataFrame({
        '날짜': data['년월'],
        '거주인구': data['거주인구']
    })
    st.line_chart(pop_chart.set_index('날짜'))
    
    # 간단한 예측 모델 구현 (이동 평균 및 선형 추세)
    st.subheader("간단한 예측 모델")
    
    # 이동 평균 기간 선택
    ma_period = st.slider("이동 평균 기간(개월)", min_value=1, max_value=12, value=3)
    
    # 이동 평균 계산
    if target_variable == "부동산 가격":
        target_col = '매매가격(만원/평)'
    else:  # "인구 변화"
        target_col = '거주인구'
    
    # 이동 평균 계산
    data[f'{target_col}_MA'] = data[target_col].rolling(window=ma_period).mean()
    
    # 선형 추세 계산
    data['time_idx'] = range(len(data))
    
    # 결측치 제거 (이동 평균의 초기 NaN 값)
    trend_data = data.dropna()
    
    # 단순 선형 관계 계산
    if len(trend_data) >= 2:
        X = trend_data['time_idx'].values
        y = trend_data[target_col].values
        
        # 선형 회귀 계수 계산 (numpy 사용)
        A = np.vstack([X, np.ones(len(X))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 추세선 추가
        data[f'{target_col}_Trend'] = data['time_idx'] * m + c
        
        st.write(f"선형 추세: {target_col} = {m:.2f} × 시간 + {c:.2f}")
        
        # 예측 결과 시각화
        pred_chart = pd.DataFrame({
            '날짜': data['년월'],
            '실제값': data[target_col],
            '이동평균': data[f'{target_col}_MA'],
            '추세선': data[f'{target_col}_Trend']
        })
        
        st.line_chart(pred_chart.set_index('날짜'))
        
        # 미래 예측
        st.subheader("미래 예측")
        months_ahead = st.slider("예측 기간(개월)", min_value=1, max_value=12, value=6)
        
        # 마지막 데이터 시점
        last_date = data['년월'].iloc[-1]
        last_idx = data['time_idx'].max()
        
        # 미래 날짜 생성
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months_ahead + 1)]
        future_idx = [last_idx + i for i in range(1, months_ahead + 1)]
        
        # 선형 추세 예측
        trend_pred = [m * idx + c for idx in future_idx]
        
        # 계절성 추가 (마지막 1년의 평균 계절 패턴 사용)
        if len(data) >= 12:
            monthly_avg = data.groupby('month')[target_col].mean()
            yearly_avg = data[target_col].mean()
            monthly_factors = monthly_avg / yearly_avg
            
            seasonal_factors = [monthly_factors.get(date.month, 1.0) for date in future_dates]
            final_pred = [trend * factor for trend, factor in zip(trend_pred, seasonal_factors)]
        else:
            final_pred = trend_pred
        
        # 미래 예측 결과
        future_df = pd.DataFrame({
            '날짜': future_dates,
            '예측값': final_pred
        })
        
        st.write("향후 예측 결과:")
        st.dataframe(future_df)
        
        # 과거 + 미래 차트
        combined_df = pd.concat([
            pred_chart[['날짜', '실제값', '추세선']],
            pd.DataFrame({
                '날짜': future_df['날짜'],
                '예측값': future_df['예측값']
            })
        ]).sort_values('날짜')
        
        # 인덱스를 날짜로 설정한 차트용 데이터프레임
        chart_df = pd.DataFrame({
            '실제값': combined_df.set_index('날짜')['실제값'],
            '추세선': combined_df.set_index('날짜')['추세선']
        })
        
        # 예측값 추가 (NaN이 아닌 값만)
        if '예측값' in combined_df.columns:
            chart_df['예측값'] = combined_df.set_index('날짜')['예측값']
        
        st.line_chart(chart_df)
        
        # 시나리오 분석
        st.subheader("시나리오 분석")
        st.write("주요 변수 변화에 따른 예측값 변화를 분석합니다.")
        
        if target_variable == "부동산 가격":
            # 인구가 부동산 가격에 미치는 영향 
            if "거주인구" in data.columns and target_col in data.columns:
                # 상관계수 계산
                corr = data['거주인구'].corr(data[target_col])
                if not np.isnan(corr):
                    # 단순 선형 관계
                    X_pop = data['거주인구'].values
                    y_price = data[target_col].values
                    
                    A_pop = np.vstack([X_pop, np.ones(len(X_pop))]).T
                    m_pop, c_pop = np.linalg.lstsq(A_pop, y_price, rcond=None)[0]
                    
                    st.write(f"인구와 부동산 가격의 관계: {target_col} = {m_pop:.6f} × 거주인구 + {c_pop:.2f}")
                    st.write(f"상관계수: {corr:.4f}")
                    
                    # 인구 변화 시나리오
                    pop_change_pct = st.slider("인구 변화율(%)", min_value=-20, max_value=20, value=10)
                    
                    # 기준값 (최근 평균)
                    base_pop = data['거주인구'].iloc[-6:].mean()
                    base_price = data[target_col].iloc[-6:].mean()
                    
                    # 변경된 인구 시나리오
                    changed_pop = base_pop * (1 + pop_change_pct/100)
                    
                    # 예측
                    changed_price = m_pop * changed_pop + c_pop
                    
                    # 변화율 계산
                    price_change = changed_price - base_price
                    price_change_pct = (price_change / base_price) * 100
                    
                    # 결과 표시
                    st.write(f"**시나리오**: 거주인구가 {pop_change_pct}% 변화할 경우")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("기준 가격", f"{base_price:.2f} 만원/평")
                    col2.metric("예상 가격", f"{changed_price:.2f} 만원/평", f"{price_change:.2f}")
                    col3.metric("변화율", f"{price_change_pct:.2f}%")
        
        else:  # "인구 변화"
            # 부동산 가격이 인구에 미치는 영향
            if "매매가격(만원/평)" in data.columns and target_col in data.columns:
                # 상관계수 계산
                corr = data['매매가격(만원/평)'].corr(data[target_col])
                if not np.isnan(corr):
                    # 단순 선형 관계
                    X_price = data['매매가격(만원/평)'].values
                    y_pop = data[target_col].values
                    
                    A_price = np.vstack([X_price, np.ones(len(X_price))]).T
                    m_price, c_price = np.linalg.lstsq(A_price, y_pop, rcond=None)[0]
                    
                    st.write(f"부동산 가격과 인구의 관계: {target_col} = {m_price:.2f} × 매매가격 + {c_price:.2f}")
                    st.write(f"상관계수: {corr:.4f}")
                    
                    # 가격 변화 시나리오
                    price_change_pct = st.slider("부동산 가격 변화율(%)", min_value=-20, max_value=20, value=-10)
                    
                    # 기준값 (최근 평균)
                    base_price = data['매매가격(만원/평)'].iloc[-6:].mean()
                    base_pop = data[target_col].iloc[-6:].mean()
                    
                    # 변경된 가격 시나리오
                    changed_price = base_price * (1 + price_change_pct/100)
                    
                    # 예측
                    changed_pop = m_price * changed_price + c_price
                    
                    # 변화율 계산
                    pop_change = changed_pop - base_pop
                    pop_change_pct = (pop_change / base_pop) * 100
                    
                    # 결과 표시
                    st.write(f"**시나리오**: 부동산 가격이 {price_change_pct}% 변화할 경우")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("기준 인구", f"{base_pop:.0f} 명")
                    col2.metric("예상 인구", f"{changed_pop:.0f} 명", f"{pop_change:.0f}")
                    col3.metric("변화율", f"{pop_change_pct:.2f}%")
        
        # 정책 시나리오 분석
        st.subheader("정책 시나리오 분석")
        
        if target_variable == "부동산 가격":
            st.write("주요 정책이 부동산 가격에 미치는 영향을 분석합니다.")
            
            policy_scenarios = {
                "주택 공급 확대 (인구 5% 증가, 소득 2% 증가)": {
                    "인구 변화율": 5,
                    "소득 변화율": 2,
                    "설명": "주택 공급이 확대되면 인구 유입이 증가하고, 지역 경제가 활성화됩니다."
                },
                "교통 인프라 개선 (인구 8% 증가, 소득 3% 증가)": {
                    "인구 변화율": 8,
                    "소득 변화율": 3,
                    "설명": "교통 접근성이 향상되면 인구 유입이 증가하고, 지역 경제가 활성화됩니다."
                },
                "상업 지구 개발 (인구 3% 증가, 소득 7% 증가)": {
                    "인구 변화율": 3,
                    "소득 변화율": 7,
                    "설명": "상업 시설 확충으로 일자리가 늘어나고 소득이 증가합니다."
                }
            }
            
            selected_policy = st.selectbox(
                "정책 시나리오 선택",
                list(policy_scenarios.keys())
            )
            
            st.write(policy_scenarios[selected_policy]["설명"])
            
            # 기준값 (최근 평균)
            base_pop = data['거주인구'].iloc[-6:].mean()
            base_price = data[target_col].iloc[-6:].mean()
            
            # 인구 영향 모델 계수 
            if "거주인구" in data.columns and target_col in data.columns:
                X_pop = data['거주인구'].values
                y_price = data[target_col].values
                A_pop = np.vstack([X_pop, np.ones(len(X_pop))]).T
                m_pop, c_pop = np.linalg.lstsq(A_pop, y_price, rcond=None)[0]
                
                # 정책 시나리오 효과 
                pop_change_pct = policy_scenarios[selected_policy]["인구 변화율"]
                income_change_pct = policy_scenarios[selected_policy]["소득 변화율"]
                
                # 인구 변화에 따른 가격 영향
                changed_pop = base_pop * (1 + pop_change_pct/100)
                pop_effect_price = m_pop * changed_pop + c_pop
                
                # 소득 효과 (단순 가정: 소득 1% 증가 시 가격 0.5% 증가)
                income_effect = base_price * (income_change_pct * 0.5 / 100)
                
                # 최종 가격 예측 (인구 효과 + 소득 효과)
                final_price = pop_effect_price + income_effect
                
                # 변화율 계산
                price_change = final_price - base_price
                price_change_pct = (price_change / base_price) * 100
                
                # 결과 표시
                col1, col2, col3 = st.columns(3)
                col1.metric("현재 가격", f"{base_price:.2f} 만원/평")
                col2.metric("예상 가격", f"{final_price:.2f} 만원/평", f"{price_change:.2f}")
                col3.metric("변화율", f"{price_change_pct:.2f}%")
                
                # 요인별 영향 분석
                st.write("### 요인별 영향")
                factor_data = pd.DataFrame({
                    '요인': ['인구 변화', '소득 변화', '총 효과'],
                    '가격 변화': [
                        pop_effect_price - base_price,
                        income_effect,
                        price_change
                    ],
                    '기여도(%)': [
                        ((pop_effect_price - base_price) / price_change) * 100 if price_change != 0 else 0,
                        (income_effect / price_change) * 100 if price_change != 0 else 0,
                        100
                    ]
                })
                
                st.dataframe(factor_data)
                
                # 시나리오 비교
                st.write("### 정책 시나리오 비교")
                scenarios_comparison = []
                
                # 기본 시나리오 추가
                scenarios_comparison.append({
                    '시나리오': '현재 상태',
                    '예측 가격': base_price,
                    '변화율(%)': 0.0
                })
                
                # 모든 정책 시나리오 추가
                for policy_name, policy_vars in policy_scenarios.items():
                    pop_change = policy_vars["인구 변화율"]
                    income_change = policy_vars["소득 변화율"]
                    
                    # 인구 변화에 따른 가격 영향
                    changed_pop = base_pop * (1 + pop_change/100)
                    pop_effect = m_pop * changed_pop + c_pop
                    
                    # 소득 효과
                    income_effect = base_price * (income_change * 0.5 / 100)
                    
                    # 최종 가격
                    policy_price = pop_effect + income_effect
                    
                    # 변화율
                    price_change_pct = ((policy_price - base_price) / base_price) * 100
                    
                    scenarios_comparison.append({
                        '시나리오': policy_name,
                        '예측 가격': policy_price,
                        '변화율(%)': price_change_pct
                    })
                
                # 정책 시나리오 비교 표시
                comparison_df = pd.DataFrame(scenarios_comparison)
                st.dataframe(comparison_df)
                
                # 시나리오 차트
                chart_df = pd.DataFrame({
                    '예측 가격': comparison_df['예측 가격']
                }, index=comparison_df['시나리오'])
                st.bar_chart(chart_df)
                
        else:  # "인구 변화"
            st.write("주택 정책이 인구 변화에 미치는 영향을 분석합니다.")
            
            housing_policy_scenarios = {
                "주택 가격 안정화 정책 (가격 10% 하락)": {
                    "가격 변화율": -10,
                    "설명": "주택 가격 안정화 정책으로 가격이 하락하면 인구 유입이 증가합니다."
                },
                "주택 공급 확대 (가격 15% 하락)": {
                    "가격 변화율": -15,
                    "설명": "주택 공급 확대로 가격이 하락하면 더 많은 인구가 유입됩니다."
                },
                "주거 환경 개선 (가격 5% 상승)": {
                    "가격 변화율": 5,
                    "설명": "주거 환경이 개선되면 가격이 상승하지만 매력도 증가로 인구 유지가 가능합니다."
                }
            }
            
            selected_housing_policy = st.selectbox(
                "주택 정책 시나리오 선택",
                list(housing_policy_scenarios.keys())
            )
            
            st.write(housing_policy_scenarios[selected_housing_policy]["설명"])
            
            # 기준값 (최근 평균)
            base_price = data['매매가격(만원/평)'].iloc[-6:].mean()
            base_pop = data[target_col].iloc[-6:].mean()
            
            # 가격 영향 모델 계수
            if "매매가격(만원/평)" in data.columns and target_col in data.columns:
                X_price = data['매매가격(만원/평)'].values
                y_pop = data[target_col].values
                A_price = np.vstack([X_price, np.ones(len(X_price))]).T
                m_price, c_price = np.linalg.lstsq(A_price, y_pop, rcond=None)[0]
                
                # 정책 시나리오 효과
                price_change_pct = housing_policy_scenarios[selected_housing_policy]["가격 변화율"]
                
                # 변경된 가격
                changed_price = base_price * (1 + price_change_pct/100)
                
                # 인구 예측
                predicted_pop = m_price * changed_price + c_price
                
                # 변화율 계산
                pop_change = predicted_pop - base_pop
                pop_change_pct = (pop_change / base_pop) * 100
                
                # 결과 표시
                col1, col2, col3 = st.columns(3)
                col1.metric("현재 인구", f"{base_pop:.0f} 명")
                col2.metric("예상 인구", f"{predicted_pop:.0f} 명", f"{pop_change:+.0f}")
                col3.metric("변화율", f"{pop_change_pct:+.2f}%")
                
                # 정책 시나리오 비교
                st.write("### 주택 정책 시나리오 비교")
                scenarios_comparison = []
                
                # 기본 시나리오 추가
                scenarios_comparison.append({
                    '시나리오': '현재 상태',
                    '인구 예측': base_pop,
                    '변화율(%)': 0.0
                })
                
                # 모든 정책 시나리오 추가
                for policy_name, policy_vars in housing_policy_scenarios.items():
                    price_change = policy_vars["가격 변화율"]
                    
                    # 변경된 가격
                    changed_price = base_price * (1 + price_change/100)
                    
                    # 인구 예측
                    predicted_pop = m_price * changed_price + c_price
                    
                    # 변화율
                    pop_change_pct = ((predicted_pop - base_pop) / base_pop) * 100
                    
                    scenarios_comparison.append({
                        '시나리오': policy_name,
                        '인구 예측': predicted_pop,
                        '변화율(%)': pop_change_pct
                    })
                
                # 정책 시나리오 비교 표시
                comparison_df = pd.DataFrame(scenarios_comparison)
                st.dataframe(comparison_df)
                
                # 시나리오 차트
                chart_df = pd.DataFrame({
                    '인구 예측': comparison_df['인구 예측']
                }, index=comparison_df['시나리오'])
                st.bar_chart(chart_df)
                
    else:
        st.warning("최소 2개 이상의 데이터 포인트가 필요합니다.")
    
    # 결론 및 인사이트
    st.subheader("결론 및 인사이트")
    st.write("""
    ### 모델 해석:
    - **이동 평균**: 단기 변동성을 제거하고 전반적인 추세를 파악할 수 있습니다.
    - **선형 추세**: 장기적인 변화 방향과 속도를 나타냅니다.
    - **시나리오 분석**: 다양한 변수 변화가 미치는 영향을 파악할 수 있습니다.

    ### 주요 인사이트:
    1. **인과관계의 방향성**: 변수 간 관계의 강도와 방향을 이해할 수 있습니다.
    2. **영향 요소**: 가격 변화와 인구 변화에 영향을 미치는 주요 요소를 확인할 수 있습니다.
    3. **민감도**: 특정 변수의 변화에 대한 민감도를 통해 정책 효과를 예측할 수 있습니다.

    ### 정책적 함의:
    - 부동산 가격과 인구 이동 간의 관계를 이해함으로써 더 효과적인 주택 정책을 설계할 수 있습니다.
    - 지역별 특성을 고려한 맞춤형 정책 접근이 필요합니다.
    """)

except Exception as e:
    st.error(f"데이터 분석 중 오류가 발생했습니다: {e}")
    st.info("필요한 데이터를 확인하고 다시 시도해주세요.")