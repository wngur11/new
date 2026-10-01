# -----------------------------
# 전체 기간 회귀분석
# -----------------------------
annual["경과연수"] = annual["연도"] - 1908

x = annual["경과연수"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)

# 100년에 몇 ℃ 변하는지
slope_100 = slope * 100

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# -----------------------------
# 최근 20년 회귀분석
# -----------------------------
recent_end_year = int(annual["연도"].max())
recent_start_year = recent_end_year - 19

recent = annual[
    (annual["연도"] >= recent_start_year) &
    (annual["연도"] <= recent_end_year)
].copy()

recent_x = recent["연도"].to_numpy(dtype=float)
recent_y = recent["연평균기온"].to_numpy(dtype=float)

recent_slope, recent_intercept = np.polyfit(
    recent_x,
    recent_y,
    1
)

# 최근 20년도 100년에 몇 ℃인지 계산
recent_slope_100 = recent_slope * 100
