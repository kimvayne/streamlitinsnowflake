import streamlit as st
from utils.data_loader import get_property_data

st.set_page_config(
    page_title="토지가격과 인구밀집도 간의 인과관계 분석",
    page_icon="🏙️",
    layout="wide"
)

st.title("토지가격과 인구밀집도 간의 인과관계 분석")
st.subheader("서울시 3개구(서초구, 영등포구, 중구) 데이터 기반 분석")

# 앱 소개
st.markdown("""
이 애플리케이션은 서울시의 서초구, 영등포구, 중구를 대상으로 토지가격과 인구밀집도 간의 인과관계를 분석합니다.

**핵심 질문**: "토지가 비싸기 때문에 사람이 많이 모이는 것인가, 아니면 사람이 많이 모이기 때문에 토지가 비싼 것인가?"

### 주요 데이터셋:
- 부동산 가격 정보: 법정동별 전세 및 매매 가격 (KOREAN_POPULATION__APARTMENT_MARKET_PRICE_DATA)
- 인구 이동 정보: 연령대/성별/시간대별 거주/방문/근무 인구 (SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS)
- 금융 및 소득 정보: 소득 분포, 자산 정보 (SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS)

### 분석 방법:
- 데이터 탐색: 기본 통계 및 시각화
- 상관관계 분석: 변수 간 관계 분석
- 인과관계 분석: 그랜저 인과관계 검정
- 예측 모델: 미래 부동산 가격 및 인구 변화 예측
""")

# Snowflake 데이터 연결 테스트
try:
    # SIS 환경에서는 st.session에 직접 접근 가능
    test_data = get_property_data(limit=5)
    if not test_data.empty:
        st.success("데이터베이스 연결에 성공했습니다.")
        st.write("데이터 샘플:")
        st.dataframe(test_data)
    else:
        st.warning("데이터베이스 연결은 성공했으나 데이터가 없습니다. 샘플 데이터를 사용합니다.")
except Exception as e:
    st.error(f"데이터베이스 연결 오류: {e}")
    st.warning("샘플 데이터로 분석을 계속합니다.")

# 사이드바 정보
st.sidebar.title("앱 탐색")
st.sidebar.info("""
**사용 방법**:
1. 왼쪽 상단의 페이지 선택기에서 원하는 분석 페이지를 선택하세요.
2. 각 페이지에서 데이터셋과 분석 옵션을 선택할 수 있습니다.
3. 분석 결과와 시각화를 탐색하세요.
""")

# 샘플 결과 미리보기
st.header("샘플 분석 결과")
col1, col2 = st.columns(2)

with col1:
    st.subheader("지역별 부동산 가격 분포")
    st.image("https://via.placeholder.com/400x300?text=Property+Price+Distribution", use_column_width=True)

with col2:
    st.subheader("시간에 따른 인구밀도 변화")
    st.image("https://via.placeholder.com/400x300?text=Population+Density+Trend", use_column_width=True)

st.markdown("---")
st.caption("© 2025 토지가격-인구밀집도 인과관계 분석 프로젝트")