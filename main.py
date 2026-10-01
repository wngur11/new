
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
st.write(
    "서울의 연도별 평균기온을 이용해 기온 변화 추세를 살펴봅니다."
)


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
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =========================================
# 2025년까지의 데이터만 사용
# =========================================
df = df[
    df["연도"] <= 2025
].copy()


# =========================================
# 연도별 평균기온 계산
# =========================================
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


# =========================================
# 관측일수가 300일 이상인 해만 사용
# =========================================
annual = annual[
    annual["관측일수"] >= 300
].copy()


# =========================================
# 1908년 이후 데이터만 사용
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
# 독립 변수:
# 1908년부터 몇 년이 지났는가
annual["경과연수"] = (
    annual["연도"] - 1908
)

x = annual["경과연수"].to_numpy(
    dtype=float
)

y = annual["연평균기온"].to_numpy(
    dtype=float
)


# 1차 회귀
slope, intercept = np.polyfit(
    x,
    y,
    1
)


# 회귀선 값
annual["회귀예상기온"] = (
    slope * annual["경과연수"]
    + intercept
)


# =========================================
# 전체 기간의 상관계수
# =========================================
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# =========================================
# 전체 기간 기울기
# ℃/년 → ℃/100년
# =========================================
slope_100 = slope * 100


# =========================================
# 최근 20년 데이터
# =========================================
start_year = int(
    annual["연도"].min()
)

end_year = int(
    annual["연도"].max()
)

recent_start_year = end_year - 19


recent = annual[
    (annual["연도"] >= recent_start_year)
    & (annual["연도"] <= end_year)
].copy()


# =========================================
# 최근 20년 회귀분석
# =========================================
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

    # ℃/년 → ℃/100년
    recent_slope_100 = (
        recent_slope * 100
    )

else:

    recent_slope_100 = np.nan


# =========================================
# 사용한 데이터 정보
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
        f"{start_year}년"
    )


with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


st.caption(
    "2025년까지의 자료 중 관측일수가 300일 이상인 해만 사용했습니다."
)


# =========================================
# ★ 100년에 몇 도 오르는가?
# =========================================
st.subheader("🌡️ 100년에 몇 도 오르는가?")


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        f"""
        <div style="
            border: 2px solid #cccccc;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        ">
            <div style="
                font-size: 20px;
                font-weight: bold;
            ">
                전체 기간
            </div>

            <div style="
                font-size: 42px;
                font-weight: bold;
                margin-top: 10px;
            ">
                {slope_100:+.2f}℃
            </div>

            <div style="
                font-size: 18px;
                margin-top: 5px;
            ">
                / 100년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div style="
            border: 2px solid #cccccc;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        ">
            <div style="
                font-size: 20px;
                font-weight: bold;
            ">
                최근 20년
            </div>

            <div style="
                font-size: 42px;
                font-weight: bold;
                margin-top: 10px;
            ">
                {recent_slope_100:+.2f}℃
            </div>

            <div style="
                font-size: 18px;
                margin-top: 5px;
            ">
                / 100년
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.caption(
    "기울기(℃/년)에 100을 곱해 100년 동안의 변화량으로 환산했습니다."
)


# =========================================
# 상관계수
# =========================================
st.subheader("📈 연도와 연평균기온의 상관관계")


st.metric(
    "상관계수",
    f"{correlation:.3f}"
)


# =========================================
# 산점도 + 전체 기간 회귀선
# =========================================
fig = go.Figure()


# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예상기온"],
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f}℃"
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
    ),

    legend=dict(
        title="구분"
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================
# 회귀식
#
