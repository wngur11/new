```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# 페이지 설정
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울 기온 데이터를 이용해 연평균기온의 변화 추세를 분석합니다.")


# 데이터 주소
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    return df


# 데이터 불러오기
try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# 2025년까지의 데이터만 사용
df = df[df["연도"] <= 2025].copy()


# 연도별 평균기온과 관측일수 계산
annual = (
    df.dropna(subset=["연도", "평균기온"])
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# 관측일수가 300일 이상인 해만 사용
annual = annual[annual["관측일수"] >= 300].copy()


# 1908년 이후의 자료만 사용
annual = annual[annual["연도"] >= 1908].copy()


# 연도 정렬
annual["연도"] = annual["연도"].astype(int)
annual = annual.sort_values("연도").reset_index(drop=True)


# 데이터가 충분한지 확인
if len(annual) < 2:
    st.error("회귀분석을 할 수 있는 데이터가 충분하지 않습니다.")
    st.stop()


# =========================================================
# 전체 기간 회귀분석
# 독립변수 = 1908년부터 지난 연수
# =========================================================

annual["경과연수"] = annual["연도"] - 1908

x = annual["경과연수"].to_numpy(dtype=float)
y = annual
```
