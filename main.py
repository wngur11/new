
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 이용하여 "
    "기온 변화 추세와 미래의 예상 기온을 살펴봅니다."
)


# ==========================================
# 데이터 주소
# ==========================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# ==========================================
# 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()

except Exception as e:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ==========================================
# 기준 기간 적용
# ==========================================
# 2025년 이후 데이터 제거
df = df[df["연도"] <= 2025].copy()


# ==========================================
# 연도별 평균기온 계산
# ==========================================
annual = (
    df.dropna(
        subset=["연도", "평균기온"]
    )
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# ==========================================
# 관측일수가 300일 이상인 해만 사용
# ==========================================
annual = annual[
    annual["관측일수"] >= 300
].copy()


# 1908년 이후 데이터만 회귀분석에 사용
annual = annual[
    annual["연도"] >= 1908
].copy()


annual["연도"] = annual["연도"].astype(int)

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


# ==========================================
# 데이터 확인
# ==========================================
if len(annual) < 2:
    st.error(
        "회귀분석을 하기 위한 데이터가 충분하지 않습니다."
    )
    st.stop()


# ==========================================
# 전체 기간 회귀분석
# ==========================================
# 1908년부터 지난 연수
annual["경과연수"] = (
    annual["연도"] - 1908
)


x = annual["경과연수"].to_numpy(
    dtype=float
)

y = annual["연평균기온"].to_numpy(
    dtype=float
)


# 1차 선형 회귀
slope, intercept = np.polyfit(
    x,
    y,
    1
)


# 전체 기간 회귀선 예상값
annual["회귀예상기온"] = (
    slope * annual["경과연수"]
    + intercept
)


# 상관계수
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# ===================================

