import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 기온의 변화 추세를 살펴보고 미래 기온을 예측합니다.")

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()

# -----------------------------
# 기준 기간 적용
# 2025년 이후 제외
# 관측일 300일 미만인 해 제외
# -----------------------------
df = df[df["연도"] <= 2025].copy()

annual = (
    df.dropna(subset=["연도", "평균기온"])
      .groupby("연도")
      .agg(
          연평균기온=("평균기온", "mean"),
          관측일수=("평균기온", "count")
      )
      .reset_index()
)

annual = annual[
    (annual["관측일수"] >= 300) &
    (annual["연도"] >= 1908)
].copy()

annual["연도"] = annual["연도"].astype(int)
annual = annual.sort_values("연도").reset_index(drop=True)

# -----------------------------
# 회귀분석
# 독립 변수 = 1908년부터 지난 연수
# -----------------------------
annual["경과연수"] = annual["연도"] - 1908

x = annual["경과연수"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

if len(annual) < 2:
    st.error("회귀분석을 하기 위한 데이터가 충분하지 않습니다.")
    st.stop()

# 1차 선형회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선의 예상값
annual["회귀예상기온"] = slope * annual["경과연수"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# -----------------------------
# 회귀선 계산 함수
# -----------------------------
def predict_temperature(year):
    elapsed_years = year - 1908
    return slope * elapsed_years + intercept


# -----------------------------
# 데이터 정보
# -----------------------------
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
data_count = len(annual)

st.subheader("📊 회귀분석에 사용한 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("사용한 해의 개수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

st.caption(
    "2025년까지의 자료 중 관측일수가 300일 이상인 해만 사용했습니다. "
    "회귀분석의 독립 변수는 '1908년부터 지난 연수'입니다."
)

# -----------------------------
# 상관계수
# -----------------------------
st.subheader("📈 연도와 연평균기온의 관계")

st.metric(
    "상관계수",
    f"{correlation:.3f}"
)

st.write(
    "상관계수는 연도와 연평균기온 사이의 선형적인 관계가 "
    "어느 정도인지를 나타냅니다."
)

# -----------------------------
# 산점도 + 회귀선
# -----------------------------
fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["회귀예상기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
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

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 회귀식
# -----------------------------
st.subheader("📐 회귀식")

st.write(
    f"**예상 연평균기온 = {slope:.4f} × (연도 - 1908) + {intercept:.4f}**"
)

st.write(
    f"연도가 1년 증가할 때 회귀선 기준으로 연평균기온은 "
    f"약 **{slope:.4f}℃** 변하는 것으로 계산됩니다."
)

# -----------------------------
# 연도 슬라이더
# -----------------------------
st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temperature = predict_temperature(selected_year)

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        background-color: #FFF4D6;
        margin: 20px 0;
    ">
        <div style="font-size: 24px; font-weight: bold;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="font-size: 52px; font-weight: bold; margin-top: 10px;">
            {predicted_temperature:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "※ 2025년 이후 값은 실제 관측값이 아니라, "
    "1908년 이후의 연평균기온으로 만든 선형 회귀 직선을 이용한 예상값입니다."
)
