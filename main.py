

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================
# 페이지 설정
# =========================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울 기온 데이터로 연평균기온의 변화 추세를 분석합니다.")


# =========================================
# 데이터 주소
# =========================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# =========================================
# 데이터 불러오기
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

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


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =========================================
# 2025년까지의 데이터만 사용
# =========================================
df = df[df["연도"] <= 2025].copy()


# =========================================
# 연도별 평균기온 계산
# =========================================
annual = (
    df.dropna(subset=["연도", "평균기온"])
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# =========================================
# 관측일수가 300일 이상인 해만 사용
# =========================================
annual = annual[
    annual["관측일수"] >= 300
].copy()


# =========================================
# 1908년 이후 자료만 사용
# =========================================
annual = annual[
    annual["연도"] >= 1908
].copy()


annual["연도"] = annual["연도"].astype(int)

annual = annual.sort_values(
    "연도"
).reset_index(drop=True)


if len(annual) < 2:
    st.error("회귀분석을 위한 데이터가 부족합니다.")
    st.stop()


# =========================================
# 전체 기간 회귀분석
# =========================================
annual["경과연수"] = (
    annual["연도"] - 1908
)

x = annual["경과연수"].to_numpy(
    dtype=float
)

y = annual["연평균기온"].to_numpy(
    dtype=float
)

slope, intercept = np.polyfit(
    x,
    y,
    1
)


# 회귀선
annual["회귀예상기온"] = (
    slope * annual["경과연수"]
    + intercept
)


# 상관계수
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# 100년 변화량
전체_100년_변화 = slope * 100


# =========================================
# 최근 20년 회귀분석
# =========================================
마지막연도 = int(
    annual["연도"].max()
)

최근20년_시작 = 마지막연도 - 19


recent = annual[
    (annual["연도"] >= 최근20년_시작)
    & (annual["연도"] <= 마지막연도)
].copy()


if len(recent) >= 2:

    recent_x = recent["연도"].to_numpy(
        dtype=float
    )

    recent_y = recent["연평균기온"].to_numpy(
        dtype=float
    )

    recent_slope, recent_intercept = np.polyfit(
        recent_x,
        recent_y,
        1
    )

    최근20년_100년_변화 = recent_slope * 100

else:

    최근20년_100년_변화 = np.nan


# =========================================
# 회귀선 정보
# =========================================
st.subheader("📊 회귀선에 사용한 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "사용한 해의 개수",
        f"{len(annual)}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(annual['연도'].min())}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{마지막연도}년"
    )

st.caption(
    "2025년까지의 자료 중 관측일수가 300일 이상인 해만 사용했습니다."
)


# =========================================
# 100년에 몇 도 변하는가
# =========================================
st.subheader("🌡️ 100년에 몇 도 오르는가?")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{전체_100년_변화:+.2f} ℃"
    )
    st.caption("100년 동안의 예상 변화량")

with col2:
    st.metric(
        f"최근 20년 ({최근20년_시작}~{마지막연도})",
        f"{최근20년_100년_변화:+.2f} ℃"
    )
    st.caption("100년으로 환산한 변화량")


# =========================================
# 상관계수
# =========================================
st.subheader("📈 상관계수")

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{correlation:.3f}"
)


# =========================================
# 산점도 + 회귀선
# =========================================
st.subheader("📉 연도별 연평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예상기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "예상기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================
# 회귀식
# =========================================
st.subheader("📐 회귀식")

st.write(
    f"예상 연평균기온 = "
    f"{slope:.4f} × (연도 - 1908) + {intercept:.4f}"
)


# =========================================
# 연도 슬라이더
# =========================================
st.subheader("🔮 연도별 예상 기온")

선택연도 = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# =========================================
# 선택한 연도의 예상 기온
# =========================================
선택연도_경과연수 = 선택연도 - 1908

예상기온 = (
    slope * 선택연도_경과연수
    + intercept
)


# =========================================
# 예상 기온
# =========================================
st.metric(
    label=f"{선택연도}년 예상 연평균기온",
    value=f"{예상기온:.2f} ℃"
)

st.caption(
    "2025년 이후 값은 실제 관측값이 아니라 "
    "전체 기간의 회귀 직선을 이용한 예상값입니다."
)

