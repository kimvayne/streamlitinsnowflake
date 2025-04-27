import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark.context import get_active_session

# SIS 환경에서 세션 가져오기
try:
    session = get_active_session()
except Exception as e:
    st.error(f"세션 초기화 오류: {e}")

@st.cache_data
def run_query(query):
    """SIS 환경에서 세션을 사용하여 SQL 쿼리 실행"""
    try:
        result = session.sql(query).collect()
        return pd.DataFrame(result)
    except Exception as e:
        st.error(f"쿼리 실행 중 오류: {e}")
        return pd.DataFrame()

@st.cache_data
def get_property_data(limit=1000, region=None):
    """부동산 가격 데이터 조회"""
    try:
        # 스노우플레이크 쿼리 작성
        where_clause = f"WHERE SGG = '{region}'" if region else ""
        query = f"""
        SELECT BJD_CODE, EMD, SD, SGG, JEONSE_PRICE_PER_SUPPLY_PYEONG, 
               MEME_PRICE_PER_SUPPLY_PYEONG, YYYYMMDD
        FROM KOREAN_POPULATION__APARTMENT_MARKET_PRICE_DATA.HACKATHON_2025Q2.REGION_APT_RICHGO_MARKET_PRICE_M_H
        {where_clause}
        LIMIT {limit}
        """
        
        result = run_query(query)
        
        if result.empty:
            print(f"No property data found for region: {region}. Using sample data.")
            return get_sample_property_data(region)
        return result
    except Exception as e:
        print(f"Error loading property data: {e}")
        return get_sample_property_data(region)

@st.cache_data
def get_population_data(limit=1000, region=None):
    """인구 이동 데이터 조회"""
    try:
        region_map = {"서초구": "11650", "영등포구": "11560", "중구": "11140"}
        district_code = region_map.get(region, "")
        where_clause = f"WHERE DISTRICT_CODE = '{district_code}'" if district_code else ""
        
        query = f"""
        SELECT DISTRICT_CODE, CITY_CODE, AGE_GROUP, GENDER, RESIDENTIAL_POPULATION, 
               VISITING_POPULATION, WORKING_POPULATION, STANDARD_YEAR_MONTH
        FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.FLOATING_POPULATION_INFO
        {where_clause}
        LIMIT {limit}
        """
        result = run_query(query)
        
        if result.empty:
            print(f"No population data found for region: {region}. Using sample data.")
            return get_sample_population_data(region)
        return result
    except Exception as e:
        print(f"Error loading population data: {e}")
        return get_sample_population_data(region)

@st.cache_data
def get_finance_data(limit=1000, region=None):
    """금융 및 소득 데이터 조회"""
    try:
        region_map = {"서초구": "11650", "영등포구": "11560", "중구": "11140"}
        district_code = region_map.get(region, "")
        where_clause = f"WHERE DISTRICT_CODE = '{district_code}'" if district_code else ""
        
        query = f"""
        SELECT DISTRICT_CODE, CITY_CODE, AGE_GROUP, GENDER, AVERAGE_INCOME, 
               AVERAGE_HOUSEHOLD_INCOME, STANDARD_YEAR_MONTH
        FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.ASSET_INCOME_INFO
        {where_clause}
        LIMIT {limit}
        """
        result = run_query(query)
        
        if result.empty:
            print(f"No finance data found for region: {region}. Using sample data.")
            return get_sample_finance_data(region)
        return result
    except Exception as e:
        print(f"Error loading finance data: {e}")
        return get_sample_finance_data(region)

@st.cache_data
def get_department_store_data(limit=1000, store_name=None):
    """백화점 방문 데이터 조회"""
    try:
        where_clause = f"WHERE DEP_NAME = '{store_name}'" if store_name else ""
        
        query = f"""
        SELECT COUNT, DATE_KST, DEP_NAME
        FROM SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_DEPARTMENT_STORE_DATA.PUBLIC.DEPARTMENT_STORE_FOOT_TRAFFIC
        {where_clause}
        LIMIT {limit}
        """
        result = run_query(query)
        
        if result.empty:
            print(f"No department store data found for store: {store_name}. Using sample data.")
            return get_sample_department_store_data(store_name)
        return result
    except Exception as e:
        print(f"Error loading department store data: {e}")
        return get_sample_department_store_data(store_name)

@st.cache_data
def get_home_office_data(limit=1000, department_store=None):
    """거주지 및 직장 위치 데이터 조회"""
    try:
        where_clause = f"WHERE DEP_NAME = '{department_store}'" if department_store else ""
        
        query = f"""
        SELECT ADDR_LV1, ADDR_LV2, ADDR_LV3, DEP_NAME, LOC_TYPE, RATIO
        FROM RESIDENTIAL__WORKPLACE_TRAFFIC_PATTERNS_FOR_SNOWFLAKE_STREAMLIT_HACKATHON.PUBLIC.SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_HOME_OFFICE_RATIO
        {where_clause}
        LIMIT {limit}
        """
        result = run_query(query)
        
        if result.empty:
            print(f"No home/office data found for store: {department_store}. Using sample data.")
            return get_sample_home_office_data(department_store)
        return result
    except Exception as e:
        print(f"Error loading home/office data: {e}")
        return get_sample_home_office_data(department_store)

# 샘플 데이터 생성 함수들
def get_sample_property_data(region=None):
    """샘플 부동산 가격 데이터 생성"""
    np.random.seed(42)
    regions = ["서초구", "영등포구", "중구"] if region is None else [region]
    
    data = []
    for r in regions:
        price_base = 3000 if r == "서초구" else (2000 if r == "영등포구" else 2500)
        jeonse_base = price_base * 0.6
        
        for i in range(333 if len(regions) > 1 else 1000):
            data.append({
                "BJD_CODE": f"{11000 + i}",
                "EMD": f"{r} 테스트동 {i%10+1}",
                "SD": "서울특별시",
                "SGG": r,
                "JEONSE_PRICE_PER_SUPPLY_PYEONG": jeonse_base + np.random.normal(0, jeonse_base*0.1),
                "MEME_PRICE_PER_SUPPLY_PYEONG": price_base + np.random.normal(0, price_base*0.1),
                "YYYYMMDD": pd.Timestamp(f"2023-{(i%12)+1:02d}-01")
            })
    
    return pd.DataFrame(data)

def get_sample_population_data(region=None):
    """샘플 인구 이동 데이터 생성"""
    np.random.seed(42)
    regions = ["서초구", "영등포구", "중구"] if region is None else [region]
    region_codes = {"서초구": "11650", "영등포구": "11560", "중구": "11140"}
    
    data = []
    for r in regions:
        pop_base = 10000 if r == "서초구" else (15000 if r == "영등포구" else 8000)
        
        for month in range(1, 13):
            for age in ["20대", "30대", "40대", "50대"]:
                for gender in ["남", "여"]:
                    data.append({
                        "DISTRICT_CODE": region_codes.get(r, "11000"),
                        "CITY_CODE": "11",
                        "AGE_GROUP": age,
                        "GENDER": gender,
                        "RESIDENTIAL_POPULATION": pop_base + np.random.normal(0, pop_base*0.05),
                        "VISITING_POPULATION": pop_base*0.4 + np.random.normal(0, pop_base*0.03),
                        "WORKING_POPULATION": pop_base*0.6 + np.random.normal(0, pop_base*0.04),
                        "STANDARD_YEAR_MONTH": f"2023{month:02d}"
                    })
    
    return pd.DataFrame(data)

def get_sample_finance_data(region=None):
    """샘플 금융 및 소득 데이터 생성"""
    np.random.seed(42)
    regions = ["서초구", "영등포구", "중구"] if region is None else [region]
    region_codes = {"서초구": "11650", "영등포구": "11560", "중구": "11140"}
    
    data = []
    for r in regions:
        income_base = 6000000 if r == "서초구" else (5000000 if r == "영등포구" else 5500000)
        
        for month in range(1, 13):
            for age in ["20대", "30대", "40대", "50대"]:
                for gender in ["남", "여"]:
                    data.append({
                        "DISTRICT_CODE": region_codes.get(r, "11000"),
                        "CITY_CODE": "11",
                        "AGE_GROUP": age,
                        "GENDER": gender,
                        "AVERAGE_INCOME": income_base + np.random.normal(0, income_base*0.1),
                        "AVERAGE_HOUSEHOLD_INCOME": income_base*1.6 + np.random.normal(0, income_base*0.15),
                        "STANDARD_YEAR_MONTH": f"2023{month:02d}"
                    })
    
    return pd.DataFrame(data)

def get_sample_department_store_data(store_name=None):
    """샘플 백화점 방문 데이터 생성"""
    np.random.seed(42)
    stores = ["롯데백화점", "현대백화점", "신세계백화점", "갤러리아"] if store_name is None else [store_name]
    
    data = []
    for store in stores:
        base_count = 5000 if store == "롯데백화점" else (4500 if store == "현대백화점" else 4000)
        
        # 2024년 1월 1일부터 30일간의 데이터 생성
        for day in range(1, 31):
            # 주말 여부에 따른 방문객 수 변동
            is_weekend = day % 7 >= 5  # 5, 6은 주말로 가정
            weekend_factor = 1.5 if is_weekend else 1.0
            
            # 날짜별 방문객 수 생성
            visit_count = int(base_count * weekend_factor + np.random.normal(0, base_count*0.1))
            
            data.append({
                "COUNT": visit_count,
                "DATE_KST": pd.Timestamp(f"2024-01-{day:02d}"),
                "DEP_NAME": store
            })
    
    return pd.DataFrame(data)

def get_sample_home_office_data(department_store=None):
    """샘플 거주지 및 직장 위치 데이터 생성"""
    np.random.seed(42)
    stores = ["롯데백화점", "현대백화점", "신세계백화점", "갤러리아"] if department_store is None else [department_store]
    districts = ["서초구", "영등포구", "중구", "강남구", "송파구"]
    loc_types = [1, 2]  # 1=집, 2=사무실
    
    data = []
    for store in stores:
        for district in districts:
            for loc_type in loc_types:
                # 비율의 합이 1이 되도록 설정
                if loc_type == 1:  # 집
                    ratio_base = 0.15 if district in ["서초구", "강남구"] else 0.05
                else:  # 사무실
                    ratio_base = 0.2 if district in ["중구", "영등포구"] else 0.1
                
                # 약간의 무작위성 추가
                ratio = ratio_base + np.random.normal(0, 0.02)
                ratio = max(0.01, min(0.5, ratio))  # 범위 제한
                
                data.append({
                    "ADDR_LV1": "서울특별시",
                    "ADDR_LV2": district,
                    "ADDR_LV3": f"{district} 테스트동",
                    "DEP_NAME": store,
                    "LOC_TYPE": loc_type,
                    "RATIO": ratio
                })
    
    return pd.DataFrame(data)