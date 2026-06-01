#%%
# -*- coding: utf-8 -*-
"""
==============================================================
  국민건강보험공단 진료내역 데이터 - 전체 EDA
==============================================================
"""
# 0. 환경 설정

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

# 환경에 맞는 한글 폰트 자동 설정
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우 (VS Code)
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')     # macOS
else:
    plt.rc('font', family='NanumBarunGothic') # 리눅스 (Colab)
plt.rcParams['axes.unicode_minus'] = False

# 기술통계 (TableOne)
try:
    from tableone import TableOne, load_dataset
except (ModuleNotFoundError, ImportError):
    # Colab 환경에서 설치 필요 시 아래 주석 해제
    # !pip install tableone
    from tableone import TableOne, load_dataset

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

print(f"현재 적용된 폰트: {plt.rcParams['font.family']}")

# 숫자를 지수 표현이 아닌 일반 표현으로 출력 / 소수점 4자리
np.set_printoptions(precision=4, suppress=True)

# 음수 기호 나오게 설정
plt.rc('axes', unicode_minus=False)

# 출력 옵션
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("패키지 로드 완료")

# %%
# 1. 데이터 로드

# data_path = "E:\Semi1_ALPACARE\data\국민건강보험공단_진료내역정보_2023.CSV.CSV"
data_path = "D:\project_semi1\국민건강보험공단_진료내역정보_2023.CSV"
# 각자 데이터 저장된 경로에 맞게 수정

raw_df = pd.read_csv(data_path, encoding="CP949")
raw_df
#%%
df = raw_df.copy()
print(f"데이터 로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")


# %%
# 2. 코드 → 레이블 매핑
age_map = {
    1: '00-04세', 2: '05-09세', 3: '10-14세', 4: '15-19세', 5: '20-24세',
    6: '25-29세', 7: '30-34세', 8: '35-39세', 9: '40-44세', 10: '45-49세',
    11: '50-54세', 12: '55-59세', 13: '60-64세', 14: '65-69세', 15: '70-74세',
    16: '75-79세', 17: '80-84세', 18: '85세이상'
}
gender_map = {1: '남자', 2: '여자'}
city_map = {
    11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주',
    30: '대전', 31: '울산', 36: '세종', 41: '경기', 42: '강원',
    43: '충북', 44: '충남', 45: '전북', 46: '전남', 47: '경북',
    48: '경남', 49: '제주'
}
form_map = {
    2: '의과입원', 3: '의과외래', 6: '조산원입원', 7: '보건기관입원',
    8: '보건기관외래', 9: '정신과낮병동', 10: '정신과입원', 11: '정신과외래'
}
dept_map = {
    0: '일반의', 1: '내과', 2: '신경과', 3: '정신과', 4: '외과',
    5: '정형외과', 6: '신경외과', 7: '흉부외과', 8: '성형외과',
    9: '마취통증', 10: '산부인과', 11: '소아청소년과', 12: '안과',
    13: '이비인후과', 14: '피부과', 15: '비뇨기과', 16: '영상의학과',
    17: '방사선종양', 18: '병리과', 19: '진단검사', 20: '결핵과',
    21: '재활의학과', 22: '핵의학과', 23: '가정의학과', 24: '응급의학과',
    80: '한방내과', 81: '한방부인과', 82: '한방소아과', 85: '침구과',
    86: '한방재활', 87: '사상체질과'
}

df['성별']    = df['성별코드'].map(gender_map)
df['연령대']  = df['연령대코드'].map(age_map)
df['시도']    = df['시도코드'].map(city_map)
df['서식']    = df['서식코드'].map(form_map)
df['진료과목'] = df['진료과목코드'].map(dept_map)
df['요양개시일자'] = pd.to_datetime(df['요양개시일자'], errors='coerce')
# df['월']      = df['요양개시일자'].dt.month
# df['분기']    = df['요양개시일자'].dt.quarter

print("레이블 매핑 완료")
print(df[['성별', '연령대', '시도', '서식', '진료과목']].head(3))


#%%

"""# ↓ EDA 주석 해재시 머신러닝 안돌아감 주의(변수가 변경됨)"""


# ═══════════════════════════════════════════════════════════════
# # ■ SECTION 1. 기본 데이터 구조 파악
# # ═══════════════════════════════════════════════════════════════

# # ─────────────────────────────────────────────────────────────
# # 1-1. 기본 정보 요약
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("1-1. 기본 정보 요약")
# print("="*60)
# print(f"총 진료 건수   : {len(df):,}건")
# print(f"고유 환자 수   : {df['가입자일련번호'].nunique():,}명")
# print(f"컬럼 수        : {df.shape[1]}개")
# print(f"기준 년도      : {df['기준년도'].unique()}")
# print(f"진료 기간      : {df['요양개시일자'].min().date()} ~ {df['요양개시일자'].max().date()}")
# df.info()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 1-2. 결측치 분석
# # ─────────────────────────────────────────────────────────────

# # Null 값 확인
# print(df.isnull().sum())

# # 국민건강정보데이터 매뉴얼 기준:
# # 부상병코드의 결측치는 'ZZ' 또는 '-' 로 표시됨
# print(len(df.loc[df["부상병코드"].str.match(r"ZZ")]))
# print(len(df.loc[df["부상병코드"].str.match(r"-")]))
# # %%
# # %%
# # ─────────────────────────────────────────────────────────────
# # 기술통계 요약 (수치형 전체)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("수치형 변수 기술통계")
# print("="*60)

# num_cols = ['요양일수', '입내원일수', '심결가산율',
#             '심결요양급여비용총액', '심결본인부담금', '심결보험자부담금', '총처방일수']
# num_cols = [c for c in num_cols if c in df.columns]

# desc = df[num_cols].describe().T
# desc['CV(%)'] = (desc['std'] / desc['mean'] * 100).round(1)   # 변동계수
# desc['왜도']  = df[num_cols].skew().round(3)
# desc['첨도']  = df[num_cols].kurt().round(3)

# # 출력 옵션 설정
# pd.set_option('display.max_columns', None)  # 모든 컬럼 출력
# pd.set_option('display.width', 2000)        # 출력 너비를 충분히 크게 설정
# pd.set_option('display.unicode.east_asian_width', True) # 한글 자출 맞춤 (중요!)

# # 기존 코드 실행...
# print(desc[['count','mean','std','min','25%','50%','75%','max','CV(%)','왜도','첨도']])


# # %%
# # ═══════════════════════════════════════════════════════════════
# # ■ SECTION 2. 범주형 변수 분포 분석
# # ═══════════════════════════════════════════════════════════════

# cat_vars = [
#     ('성별',    '성별 분포'),
#     ('연령대',  '연령대 분포'),
#     ('시도',    '시도별 분포'),
#     ('서식',    '서식(입원/외래) 분포'),
#     ('진료과목', '진료과목 분포'),
# ]

# # %%
# # ─────────────────────────────────────────────────────────────
# # 2-1. 성별 분포
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("성별 분포")
# print("="*60)
# gender_vc = df['성별'].value_counts()
# print(gender_vc)
# print(f"\n비율: {(gender_vc / len(df) * 100).round(1).to_dict()}")

# fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# # 막대그래프
# axes[0].bar(gender_vc.index, gender_vc.values,
#             color=['#5B9BD5', '#ED7D31'], edgecolor='white', width=0.5)
# axes[0].set_title('성별 진료 건수', fontsize=13, fontweight='bold')
# axes[0].set_ylabel('진료 건수')
# for i, v in enumerate(gender_vc.values):
#     axes[0].text(i, v + 0.3, f'{v:,}건\n({v/len(df)*100:.1f}%)',
#                  ha='center', fontsize=10)

# # 파이차트
# axes[1].pie(gender_vc.values, labels=gender_vc.index, autopct='%1.1f%%',
#             colors=['#5B9BD5', '#ED7D31'], startangle=90,
#             explode=(0.05, 0), textprops={'fontsize': 11})
# axes[1].set_title('성별 비율', fontsize=13, fontweight='bold')

# plt.suptitle(f'성별 분포  (총 {len(df):,}건)', fontsize=14, y=1.01)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 2-2. 연령대 분포
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("2-2. 연령대 분포")
# print("="*60)

# age_order = [age_map[k] for k in sorted(age_map.keys()) if age_map[k] in df['연령대'].unique()]
# age_vc    = df['연령대'].value_counts().reindex(age_order).dropna()
# print(age_vc)

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# palette = sns.color_palette('mako_r', n_colors=len(age_vc))

# # 막대
# bars = axes[0].bar(range(len(age_vc)), age_vc.values, color=palette, edgecolor='white')
# axes[0].set_xticks(range(len(age_vc)))
# axes[0].set_xticklabels(age_vc.index, rotation=45, ha='right')
# axes[0].set_title('연령대별 진료 건수', fontsize=13, fontweight='bold')
# axes[0].set_ylabel('진료 건수')

# # 누적 비율 꺾은선
# ax2 = axes[0].twinx()
# cum_pct = age_vc.cumsum() / age_vc.sum() * 100
# ax2.plot(range(len(age_vc)), cum_pct.values, 'r--o', markersize=4, label='누적 비율')
# ax2.set_ylabel('누적 비율 (%)', color='red')
# ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_ylim(0, 110)

# # 수평 막대
# axes[1].barh(age_vc.index[::-1], age_vc.values[::-1], color=palette[::-1], edgecolor='white')
# axes[1].set_title('연령대별 진료 건수 (수평)', fontsize=13, fontweight='bold')
# axes[1].set_xlabel('진료 건수')
# for i, v in enumerate(age_vc.values[::-1]):
#     axes[1].text(v + 0.1, i, f'{v:,}건', va='center', fontsize=9)

# plt.suptitle('연령대 분포', fontsize=14, y=1.01)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 2-3. 시도별 분포
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("2-3. 시도별 분포")
# print("="*60)

# city_vc = df['시도'].value_counts()
# print(city_vc)

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # 막대
# colors = sns.color_palette('Set2', n_colors=len(city_vc))
# axes[0].bar(city_vc.index, city_vc.values, color=colors, edgecolor='white')
# axes[0].set_title('시도별 진료 건수', fontsize=13, fontweight='bold')
# axes[0].set_ylabel('진료 건수')
# axes[0].tick_params(axis='x', rotation=45)

# # 파이
# axes[1].pie(city_vc.values, labels=city_vc.index, autopct='%1.1f%%',
#             colors=colors, startangle=140, textprops={'fontsize': 9})
# axes[1].set_title('시도별 비율', fontsize=13, fontweight='bold')

# plt.suptitle('시도별 분포', fontsize=14, y=1.01)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 2-4. 서식(입원/외래) 분포
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("2-4. 서식(입원/외래) 분포")
# print("="*60)
# form_vc = df['서식'].value_counts()
# print(form_vc)

# fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# colors_f = sns.color_palette('pastel', n_colors=len(form_vc))
# axes[0].bar(form_vc.index, form_vc.values, color=colors_f, edgecolor='white')
# axes[0].set_title('서식별 진료 건수', fontsize=13, fontweight='bold')
# axes[0].set_ylabel('진료 건수')
# axes[0].tick_params(axis='x', rotation=30)

# axes[1].pie(form_vc.values, labels=form_vc.index, autopct='%1.1f%%',
#             colors=colors_f, startangle=90, textprops={'fontsize': 10})
# axes[1].set_title('서식별 비율', fontsize=13, fontweight='bold')
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 2-5. 진료과목 분포
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("2-5. 진료과목 분포")
# print("="*60)
# dept_vc = df['진료과목'].value_counts()
# print(dept_vc)

# fig, ax = plt.subplots(figsize=(12, 5))
# colors_d = sns.color_palette('tab20', n_colors=len(dept_vc))
# bars = ax.bar(dept_vc.index, dept_vc.values, color=colors_d, edgecolor='white')
# ax.set_title('진료과목별 진료 건수', fontsize=13, fontweight='bold')
# ax.set_ylabel('진료 건수')
# ax.tick_params(axis='x', rotation=45)
# for p in bars:
#     ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.1,
#             f'{int(p.get_height()):,}', ha='center', fontsize=8)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 2-6. 주상병코드 / 부상병코드 분포 (상위 20개)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("2-6. 상병코드 분포")
# print("="*60)

# fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# for ax, col, title in zip(axes,
#                            ['주상병코드', '부상병코드'],
#                            ['주상병코드 상위 20', '부상병코드 상위 20']):
#     top = df[col].value_counts().head(20)
#     ax.barh(top.index[::-1], top.values[::-1],
#             color=sns.color_palette('Blues_r', n_colors=20), edgecolor='white')
#     ax.set_title(title, fontsize=13, fontweight='bold')
#     ax.set_xlabel('진료 건수')
#     for i, v in enumerate(top.values[::-1]):
#         ax.text(v + 0.1, i, f'{v:,}건', va='center', fontsize=8)

# plt.tight_layout()
# plt.show()

# # 상병코드 앞 3자리(대분류) 집계
# df['주상병_대분류'] = df['주상병코드'].str[:3]
# print("\n주상병 대분류 상위 10:")
# print(df['주상병_대분류'].value_counts().head(10))


# # %%
# # ═══════════════════════════════════════════════════════════════
# # ■ SECTION 3. 수치형 변수 분포 분석
# # ═══════════════════════════════════════════════════════════════

# # ─────────────────────────────────────────────────────────────
# # 3-1. 박스플롯 (수치형 전체)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("3-2. 수치형 변수 박스플롯")
# print("="*60)

# fig, axes = plt.subplots(2, 4, figsize=(18, 9))
# axes = axes.flatten()
# colors_box = sns.color_palette('Set3', n_colors=len(num_cols))

# for i, col in enumerate(num_cols):
#     ax = axes[i]
#     data = df[col].dropna()
#     bp = ax.boxplot(data, patch_artist=True, notch=False,
#                     boxprops=dict(facecolor=colors_box[i], alpha=0.7),
#                     medianprops=dict(color='crimson', linewidth=2),
#                     whiskerprops=dict(linewidth=1.5),
#                     flierprops=dict(marker='o', markersize=4, alpha=0.5))
#     ax.set_title(col, fontsize=11, fontweight='bold')

#     Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
#     IQR = Q3 - Q1
#     outlier_n = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
#     ax.set_xlabel(f'이상치: {outlier_n}건', fontsize=9)

# for j in range(len(num_cols), len(axes)):
#     axes[j].set_visible(False)

# plt.suptitle('수치형 변수 박스플롯 (이상치 포함)', fontsize=14, fontweight='bold', y=1.01)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 3-2. 이상치 상세 분석 (IQR 기준)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("3-3. 이상치 분석 (IQR 방법)")
# print("="*60)

# outlier_summary = []
# for col in num_cols:
#     data = df[col].dropna()
#     Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
#     IQR    = Q3 - Q1
#     lower  = Q1 - 1.5 * IQR
#     upper  = Q3 + 1.5 * IQR
#     n_out  = ((data < lower) | (data > upper)).sum()
#     outlier_summary.append({
#         '변수': col, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
#         '하한': lower, '상한': upper,
#         '이상치 수': n_out, '이상치 비율(%)': round(n_out/len(data)*100, 2)
#     })

# out_df = pd.DataFrame(outlier_summary)
# print(out_df.to_string(index=False))

# # 이상치 비율 시각화
# fig, ax = plt.subplots(figsize=(10, 4))
# ax.bar(out_df['변수'], out_df['이상치 비율(%)'], color='tomato', edgecolor='white')
# ax.set_title('변수별 이상치 비율 (IQR 기준)', fontsize=13, fontweight='bold')
# ax.set_ylabel('이상치 비율 (%)')
# ax.tick_params(axis='x', rotation=30)
# for i, v in enumerate(out_df['이상치 비율(%)']):
#     ax.text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=9)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 3-3. 정규성 검정 (Shapiro-Wilk / D'Agostino)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("3-4. 정규성 검정")
# print("="*60)

# normality_results = []
# for col in num_cols:
#     data = df[col].dropna()
#     # 샘플이 5000 이하이면 Shapiro-Wilk, 초과면 D'Agostino
#     if len(data) <= 5000:
#         stat, p = shapiro(data)
#         test_name = "Shapiro-Wilk"
#     else:
#         stat, p = normaltest(data)
#         test_name = "D'Agostino"
#     normality_results.append({
#         '변수': col, '검정 방법': test_name,
#         '통계량': round(stat, 4), 'p-value': round(p, 4),
#         '정규성 여부': '정규분포 ✅' if p > 0.05 else '비정규분포 ❌'
#     })

# print(pd.DataFrame(normality_results).to_string(index=False))
# #%%
# # ─────────────────────────────────────────────────────────────
# # 3-4. 총처방일수 구간 분류 (처방강도)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("3-6. 총처방일수 구간 분류")
# print("="*60)

# bins   = [-1, 3, 14, 9999]
# labels = ['단기(0-3일)', '중기(4-14일)', '장기(15일+)']
# df['처방강도'] = pd.cut(df['총처방일수'], bins=bins, labels=labels)
# presc_vc = df['처방강도'].value_counts().sort_index()
# print(presc_vc)

# fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# axes[0].bar(presc_vc.index, presc_vc.values,
#             color=['#A8D8EA', '#AA96DA', '#FCBAD3'], edgecolor='white')
# axes[0].set_title('처방강도별 분포', fontsize=13, fontweight='bold')
# axes[0].set_ylabel('진료 건수')
# for i, v in enumerate(presc_vc.values):
#     axes[0].text(i, v + 0.1, f'{v:,}건\n({v/len(df)*100:.1f}%)', ha='center', fontsize=10)

# axes[1].pie(presc_vc.values, labels=presc_vc.index, autopct='%1.1f%%',
#             colors=['#A8D8EA', '#AA96DA', '#FCBAD3'], startangle=90)
# axes[1].set_title('처방강도 비율', fontsize=13, fontweight='bold')
# plt.tight_layout()
# plt.show()


# # %%
# # ═══════════════════════════════════════════════════════════════
# # ■ SECTION 4. 시계열 분석
# # ═══════════════════════════════════════════════════════════════

# # ─────────────────────────────────────────────────────────────
# # 4-1. 월별 진료 건수 추이
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("4-1. 월별 진료 건수 추이")
# print("="*60)

# monthly = df.groupby('월').size().reset_index(name='진료건수')
# print(monthly)

# fig, axes = plt.subplots(1, 2, figsize=(13, 4))
# axes[0].plot(monthly['월'], monthly['진료건수'], 'o-', color='steelblue',
#              linewidth=2, markersize=8)
# axes[0].fill_between(monthly['월'], monthly['진료건수'], alpha=0.15, color='steelblue')
# axes[0].set_title('월별 진료 건수 추이', fontsize=13, fontweight='bold')
# axes[0].set_xlabel('월')
# axes[0].set_ylabel('진료 건수')
# axes[0].set_xticks(monthly['월'])
# for _, row in monthly.iterrows():
#     axes[0].annotate(f'{int(row["진료건수"]):,}',
#                      (row['월'], row['진료건수']), textcoords='offset points',
#                      xytext=(0, 8), ha='center', fontsize=10)

# axes[1].bar(monthly['월'], monthly['진료건수'], color='steelblue', edgecolor='white')
# axes[1].set_title('월별 진료 건수 (막대)', fontsize=13, fontweight='bold')
# axes[1].set_xlabel('월')
# axes[1].set_ylabel('진료 건수')

# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 4-2. 요일별 진료 건수
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("4-2. 요일별 진료 건수")
# print("="*60)

# dow_map  = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
# df['요일'] = df['요양개시일자'].dt.dayofweek.map(dow_map)
# dow_order  = ['월', '화', '수', '목', '금', '토', '일']
# dow_vc     = df['요일'].value_counts().reindex(dow_order).fillna(0)
# print(dow_vc)

# fig, ax = plt.subplots(figsize=(9, 4))
# colors_dow = ['#5B9BD5'] * 5 + ['#ED7D31', '#ED7D31']
# ax.bar(dow_vc.index, dow_vc.values, color=colors_dow, edgecolor='white')
# ax.set_title('요일별 진료 건수', fontsize=13, fontweight='bold')
# ax.set_ylabel('진료 건수')
# for i, v in enumerate(dow_vc.values):
#     ax.text(i, v + 0.1, f'{int(v):,}', ha='center', fontsize=10)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 4-3. 분기별 진료 건수 및 평균 처방일수
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("4-3. 분기별 집계")
# print("="*60)

# quarterly = df.groupby('분기').agg(
#     진료건수=('진료내역일련번호', 'count'),
#     평균처방일수=('총처방일수', 'mean'),
#     평균진료비=('심결요양급여비용총액', 'mean')
# ).reset_index()
# print(quarterly)

# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# for ax, col, title, color in zip(axes,
#                                   ['진료건수', '평균처방일수', '평균진료비'],
#                                   ['분기별 진료 건수', '분기별 평균 처방일수', '분기별 평균 진료비'],
#                                   ['steelblue', 'darkorange', 'seagreen']):
#     ax.bar(quarterly['분기'].astype(str) + 'Q', quarterly[col],
#            color=color, edgecolor='white', alpha=0.85)
#     ax.set_title(title, fontsize=12, fontweight='bold')
#     ax.set_ylabel(col)

# plt.suptitle('분기별 현황', fontsize=14, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 4-4. 날짜별 일 진료 건수 추이 (Time Series)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("4-4. 일별 진료 건수 추이")
# print("="*60)

# daily = df.groupby('요양개시일자').size().reset_index(name='건수')
# print(f"일별 평균 진료 건수: {daily['건수'].mean():.1f}")

# fig, ax = plt.subplots(figsize=(14, 4))
# ax.plot(daily['요양개시일자'], daily['건수'], color='steelblue', linewidth=1, alpha=0.8)
# ax.fill_between(daily['요양개시일자'], daily['건수'], alpha=0.1, color='steelblue')

# # 7일 이동평균
# daily['MA7'] = daily['건수'].rolling(7, min_periods=1).mean()
# ax.plot(daily['요양개시일자'], daily['MA7'], color='crimson', linewidth=2, label='7일 이동평균')
# ax.set_title('일별 진료 건수 추이', fontsize=13, fontweight='bold')
# ax.set_xlabel('날짜')
# ax.set_ylabel('진료 건수')
# ax.legend()
# plt.tight_layout()
# plt.show()


# # %%
# # ═══════════════════════════════════════════════════════════════
# # ■ SECTION 5. 교차 분석 (범주형 × 범주형)
# # ═══════════════════════════════════════════════════════════════
# # ─────────────────────────────────────────────────────────────
# # 5-1. 성별 × 연령대 교차표
# # ─────────────────────────────────────────────────────────────

# # 1. 연령대 매핑 및 적용
# age_map = {
#     1: '00-04세', 2: '05-09세', 3: '10-14세', 4: '15-19세', 5: '20-24세',
#     6: '25-29세', 7: '30-34세', 8: '35-39세', 9: '40-44세', 10: '45-49세',
#     11: '50-54세', 12: '55-59세', 13: '60-64세', 14: '65-69세', 15: '70-74세',
#     16: '75-79세', 17: '80-84세', 18: '85세이상'
# }
# df['연령대'] = df['연령대코드'].map(age_map)

# # 2. 데이터 준비 및 정렬
# age_order = sorted(df['연령대'].unique())
# ct_plot = pd.crosstab(df['연령대'], df['성별']).reindex(age_order)

# # 3. 그래프 그리기 (오른쪽 빈 칸 제거)
# if not ct_plot.empty:
#     # figsize를 (10, 6) 정도로 조절하면 한 개의 그래프를 보기에 적당합니다.
#     fig, ax = plt.subplots(figsize=(12, 6)) 
    
#     ct_plot.plot(kind='bar', ax=ax, color=['#5B9BD5', '#ED7D31'], 
#                  edgecolor='white', width=0.7)
    
#     ax.set_title('연령대별 성별 분포 (건수)', fontsize=14, fontweight='bold', pad=20)
#     ax.set_xlabel('연령대')
#     ax.set_ylabel('건수')
#     ax.legend(title='성별')
#     plt.xticks(rotation=45) # 연령대 라벨 기울기
    
#     plt.tight_layout()
#     plt.show()
# else:
#     print("그래프를 그릴 데이터가 없습니다.")

# # 4. 카이제곱 검정 출력
# ct_chi = pd.crosstab(df['연령대'], df['성별'])
# if ct_chi.shape[0] > 1 and ct_chi.shape[1] > 1:
#     chi2, p, dof, expected = chi2_contingency(ct_chi)
#     print("\n" + "="*50)
#     print(f"카이제곱 검정 결과")
#     print("-"*50)
#     print(f"χ²(통계량): {chi2:.4f}")
#     print(f"p-value: {p:.4f}")
#     print(f"결론: {'성별과 연령대 간 유의한 관계 있음 ✅' if p < 0.05 else '성별과 연령대 간 유의한 관계 없음 ❌'}")
#     print("="*50)

# # %%
# # ─────────────────────────────────────────────────────────────
# # 5-2. 성별 × 시도 교차 분석
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("5-2. 성별 × 시도 교차 분석")
# print("="*60)

# ct_city_gender = pd.crosstab(df['시도'], df['성별'])
# print(ct_city_gender)

# fig, ax = plt.subplots(figsize=(12, 5))
# ct_city_gender.plot(kind='bar', ax=ax, color=['#5B9BD5', '#ED7D31'],
#                     edgecolor='white', width=0.7)
# ax.set_title('시도별 성별 진료 건수', fontsize=13, fontweight='bold')
# ax.set_xlabel('')
# ax.tick_params(axis='x', rotation=45)
# ax.legend(title='성별')
# plt.tight_layout()
# plt.show()

# # 카이제곱
# if ct_city_gender.shape[0] > 1:
#     chi2, p, dof, _ = chi2_contingency(ct_city_gender)
#     print(f"카이제곱 검정: χ²={chi2:.4f}, p={p:.4f}")


# # %%
# # ─────────────────────────────────────────────────────────────
# # 5-3. 진료과목 × 성별 교차 (히트맵)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("5-3. 진료과목 × 성별 히트맵")
# print("="*60)

# ct_dept_gender = pd.crosstab(df['진료과목'], df['성별'])
# ct_dept_pct    = ct_dept_gender.div(ct_dept_gender.sum(axis=1), axis=0) * 100

# fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(ct_dept_gender)*0.5)))
# sns.heatmap(ct_dept_gender, annot=True, fmt='d', cmap='Blues',
#             ax=axes[0], linewidths=0.5)
# axes[0].set_title('진료과목 × 성별 (건수)', fontsize=12, fontweight='bold')

# sns.heatmap(ct_dept_pct.round(1), annot=True, fmt='.1f', cmap='RdYlGn',
#             ax=axes[1], linewidths=0.5, vmin=0, vmax=100)
# axes[1].set_title('진료과목 × 성별 (비율 %)', fontsize=12, fontweight='bold')
# plt.tight_layout()
# plt.show()


# # %%
# # ═══════════════════════════════════════════════════════════════
# # ■ SECTION 6. 수치형 × 범주형 그룹 비교
# # ═══════════════════════════════════════════════════════════════
# # ─────────────────────────────────────────────────────────────
# # 6-1. 시도별 총처방일수 비교 (박스플롯 + Kruskal-Wallis)
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("6-2. 시도별 총처방일수 비교")
# print("="*60)

# city_order = df.groupby('시도')['총처방일수'].median().sort_values(ascending=False).index

# fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# sns.boxplot(x='시도', y='총처방일수', data=df, order=city_order,
#             palette='viridis', ax=axes[0])
# axes[0].set_title('시도별 총처방일수 분포', fontsize=12, fontweight='bold')
# axes[0].tick_params(axis='x', rotation=45)

# city_mean = df.groupby('시도')['총처방일수'].mean().reindex(city_order)
# axes[1].bar(range(len(city_mean)), city_mean.values,
#             color=sns.color_palette('viridis', n_colors=len(city_mean)), edgecolor='white')
# axes[1].set_xticks(range(len(city_mean)))
# axes[1].set_xticklabels(city_mean.index, rotation=45)
# axes[1].set_title('시도별 평균 총처방일수', fontsize=12, fontweight='bold')
# axes[1].set_ylabel('평균 처방일수')

# plt.tight_layout()
# plt.show()

# # Kruskal-Wallis 검정
# city_groups = [df[df['시도'] == c]['총처방일수'].dropna() for c in df['시도'].unique()]
# city_groups = [g for g in city_groups if len(g) > 0]
# if len(city_groups) >= 2:
#     stat, p = kruskal(*city_groups)
#     print(f"Kruskal-Wallis 검정: H={stat:.4f}, p={p:.4f}")
#     print("→ 시도 간 처방일수 차이", "유의함 ✅" if p < 0.05 else "유의하지 않음 ❌")

# # %%
# # ═══════════════════════════════════════════════════════════════
# # ■ SECTION 7. 진료비 분석
# # ═══════════════════════════════════════════════════════════════

# # ─────────────────────────────────────────────────────────────
# # 7-1. 진료비 구성 비율 분석
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("8-1. 진료비 구성 비율 분석")
# print("="*60)

# cost_cols = ['심결본인부담금', '심결보험자부담금']
# cost_cols  = [c for c in cost_cols if c in df.columns]

# if cost_cols:
#     total_cost = df[cost_cols].sum()
#     print(total_cost)
#     print(f"\n본인부담 비율: {total_cost['심결본인부담금'] / total_cost.sum() * 100:.1f}%" if '심결본인부담금' in total_cost else "")

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # 전체 파이
#     axes[0].pie(total_cost.values, labels=['본인부담금', '보험자부담금'],
#                 autopct='%1.1f%%', colors=['#FF9999', '#66B3FF'],
#                 startangle=90, explode=(0.05, 0))
#     axes[0].set_title('전체 진료비 구성', fontsize=13, fontweight='bold')

#     # 성별별 평균 진료비 누적 막대
#     cost_gender = df.groupby('성별')[cost_cols].mean()
#     cost_gender.plot(kind='bar', stacked=True, ax=axes[1],
#                      color=['#FF9999', '#66B3FF'], edgecolor='white')
#     axes[1].set_title('성별별 평균 진료비 구성', fontsize=13, fontweight='bold')
#     axes[1].set_ylabel('평균 비용 (원)')
#     axes[1].tick_params(axis='x', rotation=0)
#     axes[1].legend(['본인부담금', '보험자부담금'])

#     plt.suptitle('진료비 구성 분석', fontsize=14, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.show()


# # %%
# # ─────────────────────────────────────────────────────────────
# # 7-2. 연령대별 평균 진료비 비교
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("8-2. 연령대별 평균 진료비 비교")
# print("="*60)

# # age_map.values()를 사용하여 실제 연령대 텍스트 리스트로 재색인합니다.
# # 만약 age_map이 리스트라면 그대로 사용하고, 딕셔너리라면 .values()를 씁니다.
# age_order = list(age_map.values()) 

# age_cost = df.groupby('연령대')['심결요양급여비용총액'].agg(['mean', 'median', 'std'])\
#              .reindex(age_order).dropna()

# print(age_cost.round(0))

# # 데이터가 있는지 확인 후 그래프 출력
# if not age_cost.empty:
#     fig, ax = plt.subplots(figsize=(12, 5))
#     x = range(len(age_cost))
#     w = 0.35
    
#     ax.bar([i - w/2 for i in x], age_cost['mean'], width=w,
#            label='평균', color='steelblue', edgecolor='white', alpha=0.85)
#     ax.bar([i + w/2 for i in x], age_cost['median'], width=w,
#            label='중앙값', color='darkorange', edgecolor='white', alpha=0.85)
    
#     ax.set_xticks(x)
#     ax.set_xticklabels(age_cost.index, rotation=45)
#     ax.set_title('연령대별 진료비 평균 vs 중앙값', fontsize=13, fontweight='bold')
#     ax.set_ylabel('진료비 (원)')
#     ax.legend()
    
#     plt.tight_layout()
#     plt.show()
# else:
#     print("출력할 데이터가 없습니다. '연령대' 컬럼의 값과 age_map의 값이 일치하는지 확인하세요.")

# %%

# # %%
# # ═══════════════════════════════════════════════════════════════
# # ■ SECTION 8. 종합 대시보드 (요약 시각화)
# # ═══════════════════════════════════════════════════════════════
# print("\n" + "="*60)
# print("10. 종합 EDA 요약 대시보드")
# print("="*60)

# fig = plt.figure(figsize=(20, 22))
# fig.suptitle('국민건강보험공단 진료내역 EDA 종합 대시보드', fontsize=18, fontweight='bold', y=0.98)

# # [1] 성별 파이
# ax1 = fig.add_subplot(4, 4, 1)
# gender_vc2 = df['성별'].value_counts()
# ax1.pie(gender_vc2.values, labels=gender_vc2.index, autopct='%1.0f%%',
#         colors=['#5B9BD5', '#ED7D31'], startangle=90)
# ax1.set_title('① 성별 비율', fontweight='bold')

# # [2] 연령대 분포
# ax2 = fig.add_subplot(4, 4, 2)
# age_vc2 = df['연령대'].value_counts().reindex(age_map)
# ax2.bar(range(len(age_vc2)), age_vc2.values,
#         color=sns.color_palette('mako_r', n_colors=len(age_vc2)), edgecolor='white')
# ax2.set_xticks(range(len(age_vc2)))
# ax2.set_xticklabels(age_vc2.index, rotation=90, fontsize=7)
# ax2.set_title('② 연령대 분포', fontweight='bold')

# # [3] 시도 분포
# ax3 = fig.add_subplot(4, 4, 3)
# city_vc2 = df['시도'].value_counts()
# ax3.barh(city_vc2.index[::-1], city_vc2.values[::-1],
#          color=sns.color_palette('Set2', n_colors=len(city_vc2))[::-1])
# ax3.set_title('③ 시도별 분포', fontweight='bold')

# # [4] 월별 추이
# ax4 = fig.add_subplot(4, 4, 4)
# monthly2 = df.groupby('월').size()
# ax4.plot(monthly2.index, monthly2.values, 'o-', color='steelblue', linewidth=2)
# ax4.fill_between(monthly2.index, monthly2.values, alpha=0.15, color='steelblue')
# ax4.set_title('④ 월별 진료 건수', fontweight='bold')
# ax4.set_xlabel('월')

# # [5] 총처방일수 히스토그램
# ax5 = fig.add_subplot(4, 4, 5)
# ax5.hist(df['총처방일수'].dropna(), bins=20, color='darkorange', edgecolor='white', alpha=0.8)
# ax5.axvline(df['총처방일수'].mean(), color='red', linestyle='--', label=f'평균 {df["총처방일수"].mean():.1f}일')
# ax5.set_title('⑤ 총처방일수 분포', fontweight='bold')
# ax5.legend(fontsize=8)

# # [6] 진료비 히스토그램
# ax6 = fig.add_subplot(4, 4, 6)
# ax6.hist(df['심결요양급여비용총액'].dropna(), bins=20, color='seagreen', edgecolor='white', alpha=0.8)
# ax6.axvline(df['심결요양급여비용총액'].mean(), color='red', linestyle='--', label=f'평균 {df["심결요양급여비용총액"].mean():,.0f}원')
# ax6.set_title('⑥ 진료비 분포', fontweight='bold')
# ax6.legend(fontsize=7)

# # [7] 처방강도 파이
# ax7 = fig.add_subplot(4, 4, 7)
# presc_vc2 = df['처방강도'].value_counts().sort_index()
# ax7.pie(presc_vc2.values, labels=presc_vc2.index, autopct='%1.0f%%',
#         colors=['#A8D8EA', '#AA96DA', '#FCBAD3'])
# ax7.set_title('⑦ 처방강도 비율', fontweight='bold')

# # [8] 요일별 분포
# ax8 = fig.add_subplot(4, 4, 8)
# dow_vc2 = df['요일'].value_counts().reindex(dow_order).fillna(0)
# colors_d = ['#5B9BD5'] * 5 + ['#ED7D31', '#ED7D31']
# ax8.bar(dow_vc2.index, dow_vc2.values, color=colors_d, edgecolor='white')
# ax8.set_title('⑧ 요일별 분포', fontweight='bold')

# # [9] 성별 × 처방강도 누적 막대
# ax9 = fig.add_subplot(4, 4, 9)
# ct9 = pd.crosstab(df['성별'], df['처방강도'], normalize='index') * 100
# ct9.plot(kind='bar', stacked=True, ax=ax9,
#          color=['#A8D8EA', '#AA96DA', '#FCBAD3'], edgecolor='white')
# ax9.set_title('⑨ 성별별 처방강도 비율', fontweight='bold')
# ax9.tick_params(axis='x', rotation=0)
# ax9.set_ylabel('%')

# # [10] 시도별 평균 처방일수
# ax10 = fig.add_subplot(4, 4, 10)
# city_presc2 = df.groupby('시도')['총처방일수'].mean().sort_values()
# ax10.barh(city_presc2.index, city_presc2.values,
#           color=sns.color_palette('viridis', n_colors=len(city_presc2)), edgecolor='white')
# ax10.set_title('⑩ 시도별 평균 처방일수', fontweight='bold')

# # [11] 상관관계 히트맵 (간략)
# ax11 = fig.add_subplot(4, 4, 11)
# mini_corr = df[['총처방일수', '요양일수', '심결요양급여비용총액', '심결본인부담금',
#                 '심결보험자부담금', '성별코드', '연령대코드']].corr()
# sns.heatmap(mini_corr, annot=True, fmt='.1f', cmap='RdYlBu_r', center=0,
#             square=True, linewidths=0.5, ax=ax11, annot_kws={'size': 7},
#             cbar=False)
# ax11.set_title('⑪ 상관관계 히트맵', fontweight='bold')
# ax11.tick_params(axis='x', rotation=45, labelsize=7)
# ax11.tick_params(axis='y', labelsize=7)

# # [12] 일별 추이 + 이동평균
# ax13 = fig.add_subplot(4, 1, 4)
# daily2 = df.groupby('요양개시일자').size().reset_index(name='건수')
# daily2['MA7'] = daily2['건수'].rolling(7, min_periods=1).mean()
# ax13.plot(daily2['요양개시일자'], daily2['건수'],
#           color='steelblue', linewidth=1, alpha=0.5, label='일별')
# ax13.plot(daily2['요양개시일자'], daily2['MA7'],
#           color='crimson', linewidth=2, label='7일 이동평균')
# ax13.fill_between(daily2['요양개시일자'], daily2['건수'], alpha=0.08, color='steelblue')
# ax13.set_title('⑬ 일별 진료 건수 추이', fontweight='bold')
# ax13.set_xlabel('날짜')
# ax13.set_ylabel('진료 건수')
# ax13.legend()

# plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.savefig('E:\Semi1_ALPACARE\data\EDA_dashboard.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("대시보드 저장 완료: EDA_dashboard.png")


# # %%
# # ─────────────────────────────────────────────────────────────
# # 최종 EDA 요약 리포트 출력
# # ─────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("  EDA 최종 요약 리포트")
# print("="*60)

# print(f"""
# ■ 데이터 개요
#   - 총 진료 건수  : {len(df):,}건
#   - 고유 환자 수  : {df['가입자일련번호'].nunique():,}명
#   - 1인당 평균 진료 건수 : {df.groupby('가입자일련번호').size().mean():.2f}건
#   - 진료 기간     : {df['요양개시일자'].min().date()} ~ {df['요양개시일자'].max().date()}

# ■ 인구학적 특성
#   - 성별 비율  : {(df['성별'].value_counts(normalize=True)*100).round(1).to_dict()}
#   - 최다 연령대: {df['연령대'].value_counts().index[0] if df['연령대'].nunique() > 0 else 'N/A'}
#   - 최다 시도  : {df['시도'].value_counts().index[0]}

# ■ 진료 특성
#   - 평균 총처방일수: {df['총처방일수'].mean():.1f}일 (중앙값 {df['총처방일수'].median():.1f}일)
#   - 장기 처방(15일+) 비율: {(df['처방강도'] == '장기(15일+)').mean()*100:.1f}%
#   - 최다 서식  : {df['서식'].value_counts().index[0] if df['서식'].nunique() > 0 else 'N/A'}
#   - 최다 과목  : {df['진료과목'].value_counts().index[0] if df['진료과목'].nunique() > 0 else 'N/A'}

# ■ 진료비
#   - 평균 총진료비 : {df['심결요양급여비용총액'].mean():,.0f}원
#   - 중앙값 총진료비: {df['심결요양급여비용총액'].median():,.0f}원
#   - 평균 본인부담금: {df['심결본인부담금'].mean():,.0f}원 ({df['심결본인부담금'].mean()/df['심결요양급여비용총액'].mean()*100:.1f}%)

# ■ 이상치
#   - 총처방일수 이상치: {((df['총처방일수'] > df['총처방일수'].quantile(0.75) + 1.5*(df['총처방일수'].quantile(0.75)-df['총처방일수'].quantile(0.25)))).sum()}건
#   - 진료비 이상치   : {((df['심결요양급여비용총액'] > df['심결요양급여비용총액'].quantile(0.75) + 1.5*(df['심결요양급여비용총액'].quantile(0.75)-df['심결요양급여비용총액'].quantile(0.25)))).sum()}건
# """)


# %%


#%%
"""
==============================================================
 PART 2. 데이터 불러오기 & 복사본 만들기
==============================================================
 2-1. 데이터 불러오기
 2-2. 복사본 만들기
==============================================================
"""

# data_path = "E:\Semi1_ALPACARE\data\국민건강보험공단_진료내역정보_2023.CSV.CSV"
# # data_path = "D:\project_semi1\국민건강보험공단_진료내역정보_2023.CSV"
# # 각자 데이터 저장된 경로에 맞게 수정

# raw_df = pd.read_csv(data_path, encoding="CP949")
# raw_df
# #%%
# # 복사본 만들기
# df=raw_df.copy()
# print("복사본 생성 완료")
# print(f"데이터 행 수: {len(df):,}건")
# #%%
# print("\n" + "="*60)
# print("1-1. 기본 정보 요약")
# print("="*60)
# print(f"총 진료 건수   : {len(df):,}건")
# print(f"고유 환자 수   : {df['가입자일련번호'].nunique():,}명")
# print(f"컬럼 수        : {df.shape[1]}개")
# print(f"기준 년도      : {df['기준년도'].unique()}")
# print(f"진료 기간      : {df['요양개시일자'].min().date()} ~ {df['요양개시일자'].max().date()}")
# df.info()


"""# ↑ EDA 주석 해재시 머신러닝 안돌아감 주의(변수가 변경됨)"""




#%%
# 국민건강정보데이터 매뉴얼 기준:
# 부상병코드의 결측치는 'ZZ' 또는 '-' 로 표시됨
print(len(df.loc[df["부상병코드"].str.match(r"ZZ")]))
print(len(df.loc[df["부상병코드"].str.match(r"-")]))

# %%
"""
==============================================================
 PART 4 전처리
==============================================================
 4-1. 기술통계 (TableOne)
 4-2. 분포 시각화
      - 수면장애 성별 / 연령대
      - 파킨슨    성별 / 연령대
      - 알츠하이머 성별 / 연령대
      - 수면세부코드별 파킨슨·알츠하이머 발병률
      - 상관관계 히트맵
      - 수면세부코드별 연령 분포 (히트맵 / 박스플롯)
==============================================================
"""
# %%
raw_df = pd.read_csv(data_path, encoding="CP949")
raw_df

df = raw_df.copy()
print(f"데이터 로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")
#%%

# """# 2. 데이터 전처리"""
# #-------------------------------------------------------------------------------------
# """## 50세 이상 추출"""

# ---------------------------------------------------------
# [추가] 50세 이상 타겟 필터링 (분석 전처리 단계)
# ---------------------------------------------------------

# df_filtered = df[df['연령대코드'] >= 11].copy()

# print(f"필터링 전 인원: {len(df)}명")
# print(f"50세 이상 인원: {len(df_filtered)}명")
# print(f"제외된 인원: {len(df) - len(df_filtered)}명")

# # 이후 분석은 50세 이상 사람들로 필터링된 df로 사용
# df = df_filtered
# ---------------------------------------------------------
#%%
"""## 2-1. 코드로 입력된 변수 처리"""
#-------------------------------------------------------------------------------------
#%%
df.info()
#%%
# 연령대코드 매핑 (1: 0~4세,..., 18: 85세 이상)
# 분석 결과의 가독성을 위해 코드 번호를 실제 연령대 명칭으로 바꿀 필요 있음
age_map = {
    1: '00-04세', 2: '05-09세', 3: '10-14세', 4: '15-19세', 5: '20-24세',
    6: '25-29세', 7: '30-34세', 8: '35-39세', 9: '40-44세', 10: '45-49세',
    11: '50-54세', 12: '55-59세', 13: '60-64세', 14: '65-69세', 15: '70-74세',
    16: '75-79세', 17: '80-84세', 18: '85세이상'
}
df['연령대'] = df['연령대코드'].map(age_map)

df['연령대'].value_counts().sort_index()
#%%
# 성별 코드 처리
df['성별'] = df['성별코드'].map({1: '남자', 2: '여자'})

df['성별'].value_counts()
#%%
# 시도코드 처리
city_map = {
    11: '서울특별시', 26: '부산광역시', 27: '대구광역시', 28: '인천광역시', 29: '광주광역시', 30: '대전광역시', 31: '울산광역시', 36: '세종특별자치시', 41: '경기도', 42: '강원도', 43: '충청북도', 44: '충청남도', 45: '전라북도', 46: '전라남도', 47: '경상북도', 48: '경상남도', 49: '제주특별자치도'
}
df['시도'] = df['시도코드'].map(city_map)

df['시도'].value_counts().sort_index()
#%%
# 서식코드 처리
# 서식코드: 입원, 외래를 구분하기 위한 분류 코드
form_map = {
    2: '의과 입원', 3: '의과 외래', 6: '조산원 입원', 7: '보건기관 입원', 8: '보건기관 외래', 9: '정신과 낮병동', 10: '정신과 입원', 11: '정신과 외래', 'ZZ': '결측', '-': '해당사항 없음'
}
df['서식'] = df['서식코드'].map(form_map)

df['서식'].value_counts().sort_index()
#%%
# 진료과목코드 처리
dept_map = {
    0: '일반의', 1: '내과', 2: '신경과', 3: '정신과', 4: '외과',
    5: '정형외과', 6: '신경외과', 7: '흉부외과', 8: '성형외과',
    9: '마취통증의학과', 10: '산부인과', 11: '소아청소년과', 12: '안과',
    13: '이비인후과', 14: '피부과', 15: '비뇨기과', 16: '영상의학과', 17: '방사선 종양학과', 18: '병리과', 19: '진단검사의학과', 20: '결핵과',
    21: '재활의학과', 22: '핵의학과', 23: '가정의학과', 24: '응급의학과', 25: '산업의학과', 26: '예방의학과', 50: '구강악안면외과', 51: '치과보철과', 52: '치과교정과', 53: '소아치과', 54: '치주과', 55: '치과보존과', 56: '구강내과', 57: '구강악안면방사선과', 58: '구강병리과', 59: '예방치과', 80: '한방내과', 81: '한방부인과', 82: '한방소아과', 83: '한방안과, 이비인', 84: '한방신경정신과', 85: '침구과', 86: '한방재활의학과', 87: '사상체질과', 88: '한방응급', 'ZZ': '결측', '-': '해당사항없음'
}

df['진료과목'] = df['진료과목코드'].map(dept_map)

df['진료과목'].value_counts()
#%%
"""## 2-2. 날짜 변환하기"""
#-------------------------------------------------------------------------------------

# 정렬 및 최초 진단일 선별을 위해 필수
df['요양개시일자'] = pd.to_datetime(df['요양개시일자'], errors='coerce')

df.info()


#%%
"""(추가)2-3. patient_df 생성 """
#-------------------------------------------------------------------------------------

# 1. 상병코드 기반 질환 여부 확인 함수
def check_disease(df, disease_codes):
    is_main = df['주상병코드'].str.startswith(disease_codes, na=False)
    is_sub = df['부상병코드'].str.startswith(disease_codes, na=False)
    return is_main | is_sub

# 2. 진단 컬럼 생성
df['알츠하이머_진단'] = check_disease(df, ('G30')).astype(int)
df['파킨슨_진단']    = check_disease(df, ('G20')).astype(int)
df['Par_Alz_진단']   = ((df['알츠하이머_진단'] == 1) | (df['파킨슨_진단'] == 1)).astype(int)
df['수면장애_진단']  = check_disease(df, ('G47')).astype(int)
df['고혈압_진단']    = check_disease(df, ('I10','I11','I12','I13','I14','I15')).astype(int)
df['지질대사_진단']  = check_disease(df, ('E78')).astype(int)
df['당뇨병_진단']    = check_disease(df, ('E11')).astype(int)
df['기관지염_진단']  = check_disease(df, ('J20')).astype(int)
df['무릎관절증_진단']= check_disease(df, ('M17')).astype(int)
df['등통증_진단']    = check_disease(df, ('M54')).astype(int)
df['성별_남성여부']  = df['성별코드'].apply(lambda x: 1 if x == 1 else 0)
df['총처방일수']     = pd.to_numeric(df['총처방일수'], errors='coerce').fillna(0)

# 3. 환자 단위로 요약 → patient_df 생성
patient_df = df.groupby('가입자일련번호').agg({
    '성별_남성여부': 'max',
    '연령대코드':   'max',
    '총처방일수':   'sum',
    '고혈압_진단':  'max',
    '지질대사_진단':'max',
    '당뇨병_진단':  'max',
    '기관지염_진단':'max',
    '무릎관절증_진단':'max',
    '등통증_진단':  'max',
    '수면장애_진단':'max',
    '알츠하이머_진단':'max',
    '파킨슨_진단':  'max',
    'Par_Alz_진단': 'max'
}).reset_index()
patient_df.rename(columns={'연령대코드': '연령'}, inplace=True)

print(f"patient_df 생성 완료: {len(patient_df)}명")

#%%
"""## (수정)3-2. 수면 영향력
수면이 알츠하이머(Alzheimer's)와 파킨슨병(Parkinson's)에 미치는 영향력이 다른 변수들(성별, 연령, 타 만성질환 등)보다 크다

* 오즈비 Odds Ratio
* 다중 로지스틱 회귀분석
* 그래프 Forest Plot
"""
#-------------------------------------------------------------------------------------

def run_regression_analysis(df, target_col):
    feature_cols = [
        '성별_남성여부', '연령', '총처방일수',
        '고혈압_진단', '지질대사_진단', '당뇨병_진단',
        '기관지염_진단', '무릎관절증_진단', '등통증_진단', '수면장애_진단'
    ]
    X = sm.add_constant(df[feature_cols].astype(float))
    y = df[target_col].astype(float)

    try:
        model = sm.Logit(y, X).fit(disp=0)
        odds_df = np.exp(model.conf_int())
        odds_df['Odds Ratio'] = np.exp(model.params)
        odds_df.columns = ['2.5%', '97.5%', 'Odds Ratio']
        return odds_df.drop('const'), model.summary()
    except Exception as e:
        return None, f"분석 불가: {e}"
#%%
#[분석 1] 로지스틱 회귀분석 (퇴행성 뇌질환 발병 영향 요인) ---

# 1. 분석할 타겟 설정
target_title = '퇴행성 뇌질환 발병 영향 요인'
target_col = 'Par_Alz_진단'

# 2. 회귀분석 실행
odds_res, summary = run_regression_analysis(patient_df, target_col)

# 3. 시각화 (하나의 그래프만 생성)
plt.figure(figsize=(8, 7))

if odds_res is not None:
    # 오즈비 기준 정렬
    odds_res = odds_res.sort_values(by='Odds Ratio', ascending=True)
    
    # 신뢰구간 에러바 계산
    errors = [
        odds_res['Odds Ratio'] - odds_res['2.5%'],
        odds_res['97.5%'] - odds_res['Odds Ratio']
    ]
    
    # 에러바 플롯 그리기
    plt.errorbar(
        odds_res['Odds Ratio'], odds_res.index,
        xerr=errors, fmt='o', color='crimson',
        capsize=5, markersize=8
    )
    
    # 기준선(OR=1) 설정
    plt.axvline(1, color='black', linestyle='--')
    
    # 타이틀 및 라벨 설정
    plt.title(f'{target_title} 발병 영향 요인\n(Odds Ratio)', fontsize=14, fontweight='bold')
    plt.xlabel('Odds Ratio (위험도)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
else:
    # 분석 불가 시 메시지 출력
    plt.text(0.5, 0.5, f'분석 불가\n{summary}', ha='center', va='center')
    plt.title(target_title)

plt.tight_layout()
plt.show()

# 4. 수치 결과 요약 출력
print(f"\n{'='*40}")
print(f"[{target_title}] 로지스틱 회귀분석 결과 요약")
print('='*40)
print(summary)
#%% 
#%%
# 다중공선성 체크: 모든 VIF 값이 5 미만으로 다중공선성 문제 없음
#VIF (Variance Inflation Factor, 분산팽창인수): 한 독립변수가 다른 독립변수들로 얼마나 잘 예측되는가
from statsmodels.stats.outliers_influence import variance_inflation_factor

feature_cols = [
    '성별_남성여부', '연령', '총처방일수',
    '고혈압_진단', '지질대사_진단', '당뇨병_진단',
    '기관지염_진단', '무릎관절증_진단', '등통증_진단', '수면장애_진단'
]

X_vif = patient_df[feature_cols].astype(float)
vif_df = pd.DataFrame({
    '변수': feature_cols,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print(vif_df.sort_values('VIF', ascending=False))
#%%
"""## 3-3. 전체 수면 추출"""
#-------------------------------------------------------------------------------------

# ---------------------------------------------------------
# [그룹 1] 수면장애(G47)가 주상병 또는 부상병에 하나라도 포함된 고유 환자
# ---------------------------------------------------------
# 원본 df에서 조건에 맞는 행 전체 추출
cond_sleep_any = (df['주상병코드'].str.contains('G47')) | \
                 (df['부상병코드'].str.contains('G47'))
sleep_any_df = df[cond_sleep_any].copy()

# 중복 제거: 가입자별로 정렬 후 가장 첫 번째(최초) 진료 기록만 유지
sleep_any_unique = sleep_any_df.sort_values(by=['가입자일련번호', '요양개시일자'])\
                               .drop_duplicates(subset=['가입자일련번호'], keep='first')

# sleep_any_df

sleep_any_unique
#%%
"""## 3-4. 전체 파킨슨 추출"""
#------------------------------------------------------------------------------------

# ---------------------------------------------------------
# [그룹 2] 파킨슨병(G20)이 주상병 또는 부상병에 하나라도 포함된 고유 환자
# ---------------------------------------------------------
# 원본 df에서 조건에 맞는 행 전체 추출
cond_parkinson_any = (df['주상병코드'].str.contains('G20')) | \
                 (df['부상병코드'].str.contains('G20'))
parkinson_any_df = df[cond_parkinson_any].copy()

#%%
parkinson_any_df

#%%

# 중복 제거: 가입자별로 정렬 후 가장 첫 번째(최초) 진료 기록만 유지
parkinson_any_unique = parkinson_any_df.sort_values(by=['가입자일련번호', '요양개시일자'])\
                               .drop_duplicates(subset=['가입자일련번호'], keep='first')
# 결과 확인
print(f"파킨슨병(G20) 고유 환자 수: {len(parkinson_any_unique)}명")

parkinson_any_unique[['가입자일련번호','성별','연령대', '주상병코드','부상병코드','총처방일수']]
#%%

"""## 3-5. 전체 알츠하이머 추출"""
#-------------------------------------------------------------------------------------

# ---------------------------------------------------------
# [그룹 3] 알츠하이머병(G30)이 주상병 또는 부상병에 하나라도 포함된 고유 환자
# ---------------------------------------------------------
# 원본 df에서 조건에 맞는 행 전체 추출
cond_alzheimer_any = (df['주상병코드'].str.contains('G30')) | \
                 (df['부상병코드'].str.contains('G30'))
alzheimer_any_df = df[cond_alzheimer_any].copy()

# 중복 제거: 가입자별로 정렬 후 가장 첫 번째(최초) 진료 기록만 유지
alzheimer_any_unique = alzheimer_any_df.sort_values(by=['가입자일련번호', '요양개시일자'])\
                               .drop_duplicates(subset=['가입자일련번호'], keep='first')
                               
alzheimer_any_unique                             
#%%
"""## 3-6. 수면 & (파킨슨 | 알츠하이머) 추출"""
#------------------------------------------------------------------------------------

# 0. 데이터 전처리: 날짜와 병명 코드 타입을 안정적으로 변환
df['요양개시일자'] = pd.to_datetime(df['요양개시일자'], errors='coerce')
df['주상병코드'] = df['주상병코드'].astype(str)
df['부상병코드'] = df['부상병코드'].astype(str)

# 1. 질환별 환자 리스트(ID) 추출
# 수면장애(G47), 파킨슨(G20), 알츠하이머(G30) 환자 고유 ID 확보
sleep_ids = df[df['주상병코드'].str.contains('G47', na=False) |
               df['부상병코드'].str.contains('G47', na=False)]['가입자일련번호'].unique()

###_family를 붙이신 의도는 아마도 **"G20(파킨슨병)이나 G30(알츠하이머병)이라는 큰 범주(Family)에 속하는 환자들"**이라는 의미로 사용
parkinson_family_ids = df[df['주상병코드'].str.contains('G20', na=False) |
                         df['부상병코드'].str.contains('G20', na=False)]['가입자일련번호'].unique()

alzheimer_family_ids = df[df['주상병코드'].str.contains('G30', na=False) |
                          df['부상병코드'].str.contains('G30', na=False)]['가입자일련번호'].unique()

# 2. 교집합 찾기 (ID 기반 매칭)
sleep_and_pd = set(sleep_ids) & set(parkinson_family_ids)
sleep_and_ad = set(sleep_ids) & set(alzheimer_family_ids)

# 3. 세부 코드 분석용 데이터 구성
sleep_history = df[df['가입자일련번호'].isin(sleep_ids)].copy()

def get_sleep_code(row):
    # 주상병/부상병에서 G47로 시작하는 코드를 찾아 앞 4자리 반환
    for c in [row['주상병코드'], row['부상병코드']]:
        if 'G47' in c:
            code_4 = c[:4]
            # 'G47' 정확히 3자리인 경우 → 데이터 오타로 판단, G470(불면증)으로 보정
            if code_4 == 'G47':
                return 'G470'
            return code_4
    return None

sleep_history['수면세부코드'] = sleep_history.apply(get_sleep_code, axis=1)

# [핵심] 환자당 하나의 대표 수면코드만 남기기 (가장 빠른 진단일 기준)
patient_sleep_map = sleep_history.dropna(subset=['수면세부코드'])\
                    .sort_values('요양개시일자')\
                    .drop_duplicates('가입자일련번호')[['가입자일련번호', '수면세부코드']]

#%%
# 4. 최종 결과 집계 및 발병 여부 마킹
patient_sleep_map['is_parkinson_family'] = patient_sleep_map['가입자일련번호'].isin(sleep_and_pd)
patient_sleep_map['is_alzheimer_family'] = patient_sleep_map['가입자일련번호'].isin(sleep_and_ad)

final_result = patient_sleep_map.groupby('수면세부코드').agg({
    'is_parkinson_family': 'sum',
    'is_alzheimer_family': 'sum',
    '가입자일련번호': 'count'
}).rename(columns={'가입자일련번호': '전체수면환자수'})

# 5. 발병률(%) 계산
final_result['파킨슨_발병률(%)'] = (final_result['is_parkinson_family'] / final_result['전체수면환자수'] * 100).round(2)
final_result['알츠하이머_발병률(%)'] = (final_result['is_alzheimer_family'] / final_result['전체수면환자수'] * 100).round(2)

print("\n[수면세부코드별 G20/G30 계열 질환 이행 분석 결과]")
print(final_result.sort_values('파킨슨_발병률(%)', ascending=False))
#%%
final_result
#%%
#%%
# 4-1. 기술통계 (TableOne)

# [핵심] 환자당 하나의 대표 수면코드만 남기기 (가장 빠른 진단일 기준)
sleep_any_unique_map = sleep_history.dropna(subset=['수면세부코드'])\
                    .sort_values('요양개시일자')\
                    .drop_duplicates('가입자일련번호')

# 수면장애이면서 파킨슨이면 1, 수면장애만 있으면 0
sleep_any_unique_map['is_parkinson'] = sleep_any_unique_map['가입자일련번호'].isin(sleep_and_pd).astype(int)

# 수면장애이면서 알츠하이머이면 1, 수면장애만 있으면 0
sleep_any_unique_map['is_alzheimer'] = sleep_any_unique_map['가입자일련번호'].isin(sleep_and_ad).astype(int)

# 수면장애이면서 파킨슨 또는 알츠하이머이면 1, 수면장애만 있으면 0
sleep_any_unique_map['is_par_alz'] = (
    sleep_any_unique_map['가입자일련번호'].isin(sleep_and_pd) |
    sleep_any_unique_map['가입자일련번호'].isin(sleep_and_ad)
).astype(int)

# %%
# 분석에 사용할 변수 선택
selected_cols = ['성별', '연령대', '시도', '서식', '진료과목', '수면세부코드',
                 '요양일수', '입내원일수', '심결가산율', '심결요양급여비용총액',
                 '심결본인부담금', '심결보험자부담금', '총처방일수',
                 'is_parkinson', 'is_alzheimer', 'is_par_alz']

df_table1 = sleep_any_unique_map[selected_cols]

# 범주형 변수
categorical = ['성별', '연령대', '시도', '서식', '진료과목', '수면세부코드',
               'is_parkinson', 'is_alzheimer', 'is_par_alz']

# 비정규분포 변수 (중앙값/사분위수 표시)
nonnormal = ['요양일수', '입내원일수', '심결요양급여비용총액',
             '심결본인부담금', '심결보험자부담금', '총처방일수']

# %%
# Example 1: 단순 요약 테이블


table1 = TableOne(df_table1, columns=selected_cols, categorical=categorical,
                  nonnormal=nonnormal, dip_test=True, normal_test=True,
                  tukey_test=True, show_histograms=True, pval=False)

print(table1.tabulate(tablefmt="github"))

# 엑셀 저장 시 아래 주석 해제
# table1.to_excel('table1_stratified_by_disease.xlsx')

# %%
# Example 2: 질환 여부(is_par_alz) 기준 층화 테이블
selected_cols2 = ['성별', '연령대', '시도', '서식', '진료과목', '수면세부코드',
                  '요양일수', '입내원일수', '심결가산율', '심결요양급여비용총액',
                  '심결본인부담금', '심결보험자부담금', '총처방일수', 'is_par_alz']
categorical2    = ['성별', '연령대', '시도', '서식', '진료과목', '수면세부코드']
nonnormal2      = ['요양일수', '입내원일수', '심결요양급여비용총액',
                   '심결본인부담금', '심결보험자부담금', '총처방일수']

table2 = TableOne(df_table1, columns=selected_cols2, categorical=categorical2,
                  nonnormal=nonnormal2, groupby='is_par_alz',
                  pval=True, htest_name=True)

print(table2.tabulate(tablefmt="github"))

# 엑셀 저장 시 아래 주석 해제
# table2.to_excel('table2_stratified_by_disease_over50.xlsx')


# %%
# 3-3. 분포 시각화
# ── [1] 수면장애 × 성별 ──────────────────────────────────────
gender_counts = sleep_any_unique['성별'].value_counts()
total_count   = len(sleep_any_unique)

plt.figure(figsize=(7, 7))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        startangle=90, colors=['#ff9999', '#66b3ff'], explode=(0.05, 0))
plt.text(0, -1.2, f'인원수 합계 = {total_count:,}명',
         ha='center', va='center', fontsize=13, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.5'))
plt.title('수면 장애에 대한 성별 비율', fontsize=15)
plt.show()

# %%
# ── [2] 수면장애 × 연령대 ─────────────────────────────────────
total_n   = len(sleep_any_unique)
age_order = sorted(sleep_any_unique['연령대'].unique())
mako_colors = sns.color_palette('mako_r', n_colors=18)

plt.figure(figsize=(12, 6))
ax = sns.countplot(data=sleep_any_unique, x='연령대', order=age_order,
                   palette=mako_colors, hue='연령대', hue_order=age_order, legend=False)
plt.text(0.25, 0.95, f'인원수 합계 = {total_n:,}명',
         transform=ax.transAxes, ha='right', va='top', fontsize=13, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
plt.title('연령대별 수면장애 분포', fontsize=15, pad=20)
plt.xlabel('연령대', fontsize=12)
plt.ylabel('환자 수(명)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# ── [3] 파킨슨 × 성별 ────────────────────────────────────────
gender_counts = parkinson_any_unique['성별'].value_counts()
total_count   = len(parkinson_any_unique)

plt.figure(figsize=(7, 7))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        startangle=90, colors=['#ff9999', '#66b3ff'], explode=(0.05, 0))
plt.text(0, -1.2, f'인원수 합계 = {total_count:,}명',
         ha='center', va='center', fontsize=13, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.5'))
plt.title('파킨슨병에 대한 성별 비율', fontsize=15)
plt.show()

# %%
# ── [4] 파킨슨 × 연령대 ───────────────────────────────────────
total_n   = len(parkinson_any_unique)
age_order = sorted(parkinson_any_unique['연령대'].unique())

plt.figure(figsize=(12, 6))
ax = sns.countplot(data=parkinson_any_unique, x='연령대', order=age_order,
                   palette='mako', hue='연령대', legend=False)
plt.text(0.25, 0.95, f'인원수 합계 = {total_n:,}명',
         transform=ax.transAxes, ha='right', va='top', fontsize=13, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
plt.title('연령대별 파킨슨병 분포', fontsize=15, pad=20)
plt.xlabel('연령대', fontsize=12)
plt.ylabel('환자 수(명)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# ── [5] 알츠하이머 × 성별 ────────────────────────────────────
gender_counts = alzheimer_any_unique['성별'].value_counts()
total_count   = len(alzheimer_any_unique)

plt.figure(figsize=(7, 7))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        startangle=90, colors=['#ff9999', '#66b3ff'], explode=(0.05, 0))
plt.text(0, -1.2, f'인원수 합계 = {total_count:,}명',
         ha='center', va='center', fontsize=13, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray', boxstyle='round,pad=0.5'))
plt.title('알츠하이머병에 대한 성별 비율', fontsize=15)
plt.show()

#%%
#정의 
# 연령대코드 매핑 (1: 0~4세,..., 18: 85세 이상)
# 분석 결과의 가독성을 위해 코드 번호를 실제 연령대 명칭으로 바꿀 필요 있음
age_map = {
    1: '00-04세', 2: '05-09세', 3: '10-14세', 4: '15-19세', 5: '20-24세',
    6: '25-29세', 7: '30-34세', 8: '35-39세', 9: '40-44세', 10: '45-49세',
    11: '50-54세', 12: '55-59세', 13: '60-64세', 14: '65-69세', 15: '70-74세',
    16: '75-79세', 17: '80-84세', 18: '85세이상'
}
df['연령대'] = df['연령대코드'].map(age_map)

df['연령대'].value_counts().sort_index()
#%%
# %%
# ── [6] 알츠하이머 × 연령대 ──────────────────────────────────
total_n   = len(alzheimer_any_unique)
age_order = sorted(alzheimer_any_unique['연령대'].unique())

plt.figure(figsize=(12, 6))
ax = sns.countplot(data=alzheimer_any_unique, x='연령대', order=age_order,
                   palette='mako', hue='연령대', legend=False)
plt.text(0.25, 0.95, f'인원수 합계 = {total_n:,}명',
         transform=ax.transAxes, ha='right', va='top', fontsize=13, fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
plt.title('연령대별 알츠하이머병 분포', fontsize=15, pad=20)
plt.xlabel('연령대', fontsize=12)
plt.ylabel('환자 수(명)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
# ── [7] 수면장애 유형별 파킨슨·알츠하이머 발병률 ─────────────

# 1. 수면장애 코드 명칭 매핑 설정
sleep_map = {
    'G470': 'G470 불면증', 
    'G471': 'G471 과다수면',
    'G472': 'G472 수면각성일정장애', 
    'G473': 'G473 수면무호흡',
    'G474': 'G474 기면증', 
    'G478': 'G478 기타수면장애',
    'G479': 'G479 상세불명수면장애'
}

# 2. 파킨슨 또는 알츠하이머 통합 발병 여부 마킹 (OR 연산)
patient_sleep_map['is_neuro_degenerative'] = (
    patient_sleep_map['가입자일련번호'].isin(sleep_and_pd) | 
    patient_sleep_map['가입자일련번호'].isin(sleep_and_ad)
).astype(int)

# 3. 수면세부코드별 최종 결과 집계
final_result_sum = patient_sleep_map.groupby('수면세부코드').agg({
    'is_neuro_degenerative': 'sum',  # 통합 환자 수 합계
    '가입자일련번호': 'count'          # 전체 환자 수
}).rename(columns={
    'is_neuro_degenerative': '퇴행성뇌질환_합계(명)', 
    '가입자일련번호': '전체수면환자수'
})

# 4. 통합 발병률(%) 계산
final_result_sum['통합_발병률(%)'] = (
    final_result_sum['퇴행성뇌질환_합계(명)'] / final_result_sum['전체수면환자수'] * 100
).round(2)

# 5. 시각화용 데이터 준비 (코드순 오름차순 정렬)
plot_data = final_result_sum.reset_index()
plot_data['수면장애유형'] = plot_data['수면세부코드'].map(sleep_map)
# '수면세부코드' 기준 오름차순 정렬
plot_data = plot_data.sort_values('수면세부코드', ascending=True)

# 6. 시각화 설정
plt.figure(figsize=(14, 8))

# 바 차트 생성 (수면세부코드 오름차순 기준)
ax = sns.barplot(
    data=plot_data, 
    x='수면장애유형', 
    y='통합_발병률(%)', 
    palette='mako'
)

# 그래프 타이틀 및 라벨 설정
plt.title('수면장애 유형별 신경퇴행성 뇌질환(파킨슨·알츠하이머) 통합 발병률', fontsize=18, pad=25)
plt.xlabel('수면 장애 유형', fontsize=13)
plt.ylabel('통합 발병률 (%)', fontsize=13)

# 세부 디자인 조정
plt.xticks(rotation=15) 
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.ylim(0, plot_data['통합_발병률(%)'].max() * 1.2)

plt.tight_layout()
plt.show()

# 데이터프레임 최종 확인
# final_result_sum.sort_index()
# %%
# ── [8] 분석변수 상관관계 히트맵 ─────────────────────────────


corr_df = sleep_any_unique_map.copy()

# 범주형 → 수치화
corr_df['성별_numeric']  = corr_df['성별'].map({'남자': 0, '여자': 1})
corr_df['연령대_numeric'] = corr_df['연령대코드']

# 수면세부코드 더미 변수 생성
sleep_codes = ['G470', 'G471', 'G472', 'G473', 'G474', 'G478', 'G479']
for code in sleep_codes:
    corr_df[code] = (corr_df['수면세부코드'] == code).astype(int)

rename_map = {
    '성별_numeric': '성별(남0,여1)', '연령대_numeric': '연령대코드',
    'G470': 'G470 불면증',          'G471': 'G471 과다수면',
    'G472': 'G472 수면각성일정장애', 'G473': 'G473 수면무호흡',
    'G474': 'G474 기면증',          'G478': 'G478 기타수면장애',
    'G479': 'G479 상세불명수면장애'
}
analysis_data = corr_df[list(rename_map.keys())].rename(columns=rename_map)
corr_matrix   = analysis_data.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": .75}, annot_kws={"size": 10})
plt.title('분석변수 상관관계 분석', fontsize=15, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%%
# ── [9] 수면세부코드별 연령대 분포 히트맵 ────────────────────
# (발병률 상위 top_n 코드 기준)

# --- 추가해야 할 부분 ---
# 예: 파킨슨 발병률이 높은 상위 10개 수면 장애 코드를 추출
top_n = 10
top_sleep_codes = final_result.sort_values('파킨슨_발병률(%)', ascending=False).head(top_n).index.tolist()

age_dist_df = pd.merge(patient_sleep_map,
                       df[['가입자일련번호', '연령대']].drop_duplicates(),
                       on='가입자일련번호', how='left')
age_distribution = age_dist_df.pivot_table(
    index='수면세부코드', columns='연령대',
    values='가입자일련번호', aggfunc='count', fill_value=0
)
age_dist_pct = (age_distribution.div(age_distribution.sum(axis=1), axis=0) * 100).round(1)
top_5_dist   = age_dist_pct.loc[top_sleep_codes].sort_index()
yticklabels  = [sleep_map.get(c, c) for c in top_5_dist.index]

plt.figure(figsize=(12, 6))
sns.heatmap(top_5_dist, annot=True, fmt=".1f", cmap="coolwarm", yticklabels=yticklabels)
plt.title("Age Distribution by Sleep Disorder Code (%)")
plt.xlabel("Age Group")
plt.ylabel("Sleep Code")
plt.show()
#%%
top_5_dist

# %%
# ── [10] 수면세부코드별 최초 진단 연령 박스플롯 ──────────────

# 1. 계산을 위한 연령대 숫자 매핑 (각 구간의 중앙값 사용) (문자열에서 숫자형으로 변환)
age_numeric_map = {
    1: 2.5,  2: 7.5,  3: 12.5, 4: 17.5, 5: 22.5,
    6: 27.5, 7: 32.5, 8: 37.5, 9: 42.5, 10: 47.5,
    11: 52.5, 12: 57.5, 13: 62.5, 14: 67.5, 15: 72.5,
    16: 77.5, 17: 82.5, 18: 87.5
}

# 2. 연령 분석용 데이터 준비
# '연령대코드'를 숫자로 매핑하여 '연령숫자' 컬럼 생성
first_visit_age = df.sort_values('요양개시일자').drop_duplicates('가입자일련번호')[['가입자일련번호', '연령대코드']]
age_analysis_df = pd.merge(patient_sleep_map, first_visit_age, on='가입자일련번호', how='left')

# 계산 가능한 숫자 컬럼을 만들기 위해 '연령숫자' 컬럼 생성
age_analysis_df['연령숫자'] = age_analysis_df['연령대코드'].map(age_numeric_map)


# 3. 수면 코드별 평균 진단 연령 통계 계산 
age_summary = age_analysis_df[age_analysis_df['수면세부코드'].isin(top_sleep_codes)].groupby('수면세부코드').agg({
    '연령숫자': 'mean'
}).reset_index().rename(columns={'연령숫자': '평균진단연령'})

plot_df = age_analysis_df[age_analysis_df['수면세부코드'].isin(top_sleep_codes)].copy()
plot_df['수면장애명'] = plot_df['수면세부코드'].map(sleep_map)

sorted_codes  = sorted(plot_df['수면세부코드'].unique())
sorted_korean = [sleep_map[c] for c in sorted_codes]

plt.figure(figsize=(14, 6))
sns.boxplot(x='수면장애명', y='연령숫자', data=plot_df, order=sorted_korean, palette='mako')

ax = plt.gca()
ax.set_xticklabels([
    f'{kor}\n({code})'
    for kor, code in zip(sorted_korean, sorted_codes)
], fontsize=10)

plt.title('주요 수면 장애별 최초 진단 연령 분포', fontsize=14, pad=15)
plt.xlabel('수면 장애 종류', fontsize=12)
plt.ylabel('연령 (추정치)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


#%%
# """# 4. EDA

# ## 4-1. 기술통계
# """
#%%
"""# 5. 통계분석

## 5-1. 수면세부코드별 최초 진단 연령 분석 코드
"""
#-------------------------------------------------------------------------------------

# #%%


#1. 수면 코드별 연령대 분포 분석 코드
# 이 코드는 각 수면 코드별로 10세 단위(또는 기존 코드 단위)로 환자가 어떻게 분포해 있는지 비율을 계산

# 1. 분석용 데이터 병합 (수면 코드 + 연령대 명칭)
age_dist_df = pd.merge(patient_sleep_map, df[['가입자일련번호', '연령대']].drop_duplicates(), on='가입자일련번호', how='left')

# 2. 피벗 테이블을 이용한 분포표 생성
# 행: 수면세부코드, 열: 연령대, 값: 환자 수
age_distribution = age_dist_df.pivot_table(
    index='수면세부코드',
    columns='연령대',
    values='가입자일련번호',
    aggfunc='count',
    fill_value=0
)

# 3. 빈도수를 비율(%)로 변환 (각 행의 합으로 나눔)
age_dist_pct = age_distribution.div(age_distribution.sum(axis=1), axis=0) * 100
age_dist_pct = age_dist_pct.round(1)

# 4. 상위 5개 수면 코드만 필터링하여 출력
top_5_dist = age_dist_pct.loc[top_sleep_codes]

print("[상위 수면세부코드별 연령대 분포 (단위: %)]")
display(top_5_dist)



#%%
# 시각화 (Heatmap)
# 수면 코드별 연령 분포

# 1. 한글 명칭으로 바꾸기
sleep_map = {     'G470': 'G470 불면증',     'G471': 'G471 과다수면',     'G472': 'G472 수면각성일정장애',     'G473': 'G473 수면무호흡',     'G474': 'G474 기면증',     'G478': 'G478 기타수면장애',     'G479': 'G479 상세불명수면장애' }

# 2. 수면 코드(y축) 오름차순으로 정렬
top_5_dist = top_5_dist.sort_index()


yticklabels = [sleep_map.get(code, code) for code in top_5_dist.index]


# 히트맵 시각화
plt.figure(figsize=(12, 6))
sns.heatmap(top_5_dist, annot=True, fmt=".1f", cmap="coolwarm",yticklabels=yticklabels)
plt.title("Age Distribution by Sleep Disorder Code (%)")
plt.xlabel("Age Group")
plt.ylabel("Sleep Code")
plt.show()
#%%
"""####히트맵 해석

"""
#%% 수면장애 상관관계 히트맵 

# 1. 분석을 위한 데이터 복사 (환자 단위 요약 데이터 사용)
# sleep_any_unique_map 기반으로 분석용 df 생성
corr_df = sleep_any_unique_map.copy()

# 2. 범주형 변수 수치화 (Label Encoding 또는 Dummy 변수 생성)
# 성별: 남자(0), 여자(1) - 이미 처리되어 있다면 확인 후 적용
corr_df['성별_numeric'] = corr_df['성별'].map({'남자': 0, '여자': 1})

# 연령대: '연령대코드' (이미 1~18 숫자로 되어 있음) 사용
corr_df['연령대_numeric'] = corr_df['연령대코드']

# 3. 요청하신 7가지 수면세부코드에 대한 더미(Dummy) 변수 생성
sleep_codes = ['G470', 'G471', 'G472', 'G473', 'G474', 'G478', 'G479']
for code in sleep_codes:
    corr_df[code] = (corr_df['수면세부코드'] == code).astype(int)

# 4. 히트맵에 들어갈 최종 컬럼 리스트 정의 및 이름 변경
# 한글 출력을 위해 딕셔너리 매핑
rename_map = {
    '성별_numeric': '성별(남0,여1)',
    '연령대_numeric': '연령대코드',
    'G470': 'G470 불면증',
    'G471': 'G471 과다수면',
    'G472': 'G472 수면각성일정장애',
    'G473': 'G473 수면무호흡',
    'G474': 'G474 기면증',
    'G478': 'G478 기타수면장애',
    'G479': 'G479 상세불명수면장애'
}

final_corr_cols = list(rename_map.keys())
analysis_data = corr_df[final_corr_cols].rename(columns=rename_map)


# 5. 분석용 데이터 준비 (이전 단계에서 생성한 analysis_data 활용)
corr_matrix = analysis_data.corr()

# 6. 시각화 설정
plt.figure(figsize=(10, 10)) # 가로 세로 비율을 동일하게 설정

# 7. 히트맵 그리기
sns.heatmap(corr_matrix, 
            annot=True,           # 숫자 표시
            fmt=".2f",            # 소수점 둘째 자리까지
            cmap='RdYlBu_r',      
            center=0,             # 0을 기준으로 색상 분산
            square=True,          # 각 셀을 정사각형으로 고정 (이미지처럼 보이게 하는 핵심)
            linewidths=0.5,       # 셀 사이의 간격
            cbar_kws={"shrink": .75}, # 컬러바 크기 조절
            annot_kws={"size": 10})    # 글자 크기 조절

plt.title('분석변수 상관관계 분석', fontsize=15, pad=20)
plt.xticks(rotation=45, ha='right') # x축 라벨 각도 조절
plt.yticks(rotation=0)              # y축 라벨은 수평으로
plt.tight_layout()
plt.show()

#%%
#시각화 상자그림
#수면 코드별 연령 분포

plt.figure(figsize=(12, 6))
sns.boxplot(x='수면세부코드', y='연령숫자', data=age_analysis_df[age_analysis_df['수면세부코드'].isin(top_sleep_codes)])
plt.title('주요 수면 장애별 최초 진단 연령 분포')
plt.ylabel('연령 (추정치)')
plt.show()


# 시각화 상자그림
# 수면 코드별 연령 분포

# 수면 코드 한글 매핑
sleep_map = {     'G470': 'G470 불면증',     'G471': 'G471 과다수면',     'G472': 'G472 수면각성일정장애',     'G473': 'G473 수면무호흡',     'G474': 'G474 기면증',     'G478': 'G478 기타수면장애',     'G479': 'G479 상세불명수면장애' }

# 분석 데이터 필터링 및 한글명 컬럼 추가
plot_df = age_analysis_df[age_analysis_df['수면세부코드'].isin(top_sleep_codes)].copy()
plot_df['수면장애명'] = plot_df['수면세부코드'].map(sleep_map)

# 오름차순 정렬 (코드 기준)
sorted_codes = sorted(plot_df['수면세부코드'].unique())
sorted_korean = [sleep_map[c] for c in sorted_codes]

plt.figure(figsize=(14, 6))
sns.boxplot(
    x='수면장애명',
    y='연령숫자',
    data=plot_df,
    order=sorted_korean,   # 코드 오름차순에 맞춘 한글 순서
    palette='mako'
)

# x축 아래에 코드명도 함께 표시
ax = plt.gca()
ax.set_xticklabels([
    f'{kor}\n({code})'
    for kor, code in zip(sorted_korean, sorted_codes)
], fontsize=10)

plt.title('주요 수면 장애별 최초 진단 연령 분포', fontsize=14, pad=15)
plt.xlabel('수면 장애 종류', fontsize=12)
plt.ylabel('연령 (추정치)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%%
#오즈비(Odds Ratio)와 p-value를 계산하는 함수가 정의
# from scipy.stats import fisher_exact

def calculate_odds_ratio_fixed(df, sleep_code, target_disease_col):
    """
    df: 분석용 데이터프레임 (patient_sleep_map)
    sleep_code: 분석할 수면 장애 코드 (예: 'G470')
    target_disease_col: 타겟 질환 컬럼명 (예: 'is_parkinson_family')
    """
    # 2x2 Table 생성
    # [ [질환O & 수면O, 질환O & 수면X], [질환X & 수면O, 질환X & 수면X] ]

    a = len(df[(df['수면세부코드'] == sleep_code) & (df[target_disease_col] == 1)])
    b = len(df[(df['수면세부코드'] != sleep_code) & (df[target_disease_col] == 1)])
    c = len(df[(df['수면세부코드'] == sleep_code) & (df[target_disease_col] == 0)])
    d = len(df[(df['수면세부코드'] != sleep_code) & (df[target_disease_col] == 0)])

    # Fisher's Exact Test로 Odds Ratio와 P-value 계산
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]])

    return odds_ratio, p_value
#%%
patient_sleep_map

#%%

# --- 추가해야 할 부분 ---
# 예: 파킨슨 발병률이 높은 상위 10개 수면 장애 코드를 추출
top_n = 10
top_sleep_codes = final_result.sort_values('파킨슨_발병률(%)', ascending=False).head(top_n).index.tolist()
# -----------------------


# 1. 계산을 위한 연령대 숫자 매핑 (각 구간의 중앙값 사용) (문자열에서 숫자형으로 변환)
age_numeric_map = {
    1: 2.5,  2: 7.5,  3: 12.5, 4: 17.5, 5: 22.5,
    6: 27.5, 7: 32.5, 8: 37.5, 9: 42.5, 10: 47.5,
    11: 52.5, 12: 57.5, 13: 62.5, 14: 67.5, 15: 72.5,
    16: 77.5, 17: 82.5, 18: 87.5
}

# 2. 연령 분석용 데이터 준비
# '연령대코드'를 숫자로 매핑하여 '연령숫자' 컬럼 생성
first_visit_age = df.sort_values('요양개시일자').drop_duplicates('가입자일련번호')[['가입자일련번호', '연령대코드']]
age_analysis_df = pd.merge(patient_sleep_map, first_visit_age, on='가입자일련번호', how='left')

# 계산 가능한 숫자 컬럼을 만들기 위해 '연령숫자' 컬럼 생성
age_analysis_df['연령숫자'] = age_analysis_df['연령대코드'].map(age_numeric_map)


# 3. 수면 코드별 평균 진단 연령 통계 계산 
age_summary = age_analysis_df[age_analysis_df['수면세부코드'].isin(top_sleep_codes)].groupby('수면세부코드').agg({
    '연령숫자': 'mean'
}).reset_index().rename(columns={'연령숫자': '평균진단연령'})

# 4. 결과 데이터 결합 (기존 final_result의 합계 데이터와 연령 데이터 병합)
# final_result에서 필요한 환자 수 컬럼들을 가져옴
final_report = pd.merge(
    final_result.loc[top_sleep_codes].reset_index(),
    age_summary,
    on='수면세부코드'
)

# 5. 컬럼명 정리 및 출력
# 분석 가독성을 위해 컬럼 이름을 직관적으로 변경하기
final_report = final_report.rename(columns={
    'is_parkinson_family': '파킨슨_환자수',
    'is_alzheimer_family': '알츠하이머_환자수',
    '전체수면환자수': '수면장애_총환자수'
})

print("[발병률 상위 수면 장애별 질환 이행 및 진단 연령 분석]")
# 수면세부코드 별로 환자 수, 발병률, 평균 연령을 모두 포함하여 출력
display(final_report[[
    '수면세부코드',
    '수면장애_총환자수',
    '파킨슨_환자수', '파킨슨_발병률(%)',
    '알츠하이머_환자수', '알츠하이머_발병률(%)',
    '평균진단연령'
]])

#%%

#%%

# 발병률 상위 10개 코드에 대해 OR과 p-value를 모두 포함한 리포트 생성
#어떤 수면 장애가 가장 위험한지
report_list = []

for code in top_sleep_codes:
    or_pd, p_pd = calculate_odds_ratio_fixed(patient_sleep_map, code, 'is_parkinson_family')
    or_ad, p_ad = calculate_odds_ratio_fixed(patient_sleep_map, code, 'is_alzheimer_family')

    report_list.append({
        '수면세부코드': code,
        '파킨슨_OR': round(or_pd, 2),
        '파킨슨_P': round(p_pd, 4),
        '알츠하이머_OR': round(or_ad, 2),
        '알츠하이머_P': round(p_ad, 4)
    })

final_stat_report = pd.DataFrame(report_list)
# 기존 final_report와 병합하여 환자 수 정보까지 합치기
total_summary = pd.merge(final_report, final_stat_report, on='수면세부코드')

# display(total_summary)



#%%
# 오즈비 결과표 수정 전
from scipy.stats import fisher_exact

# 1. [함수 정의] 오즈비와 p-value를 동시에 계산하는 안전한 함수
def get_statistical_metrics(df_map, sleep_code, target_col):
    # 해당 수면 코드가 있는지와 대상 질환 여부 필터링
    has_sleep = (df_map['수면세부코드'] == sleep_code)
    has_disease = (df_map[target_col] == True)

    # 2x2 교차표 구성 (a:노출O/질환O, b:노출O/질환X, c:노출X/질환O, d:노출X/질환X)
    a = len(df_map[has_sleep & has_disease])
    b = len(df_map[has_sleep & ~has_disease])
    c = len(df_map[~has_sleep & has_disease])
    d = len(df_map[~has_sleep & ~has_disease])

    # Fisher's Exact Test (데이터 수가 적을 수 있으므로 카이제곱보다 안전함)
    try:
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
    except:
        odds_ratio, p_value = np.nan, np.nan

    return odds_ratio, p_value

# 2. [데이터 취합] 상위 5개 수면코드에 대해 루프를 돌며 결과 저장
stats_results = []

for code in top_sleep_codes:
    # 파킨슨 통계
    or_pd, p_pd = get_statistical_metrics(patient_sleep_map, code, 'is_parkinson_family')
    # 알츠하이머 통계
    or_ad, p_ad = get_statistical_metrics(patient_sleep_map, code, 'is_alzheimer_family')

    stats_results.append({
        '수면세부코드': code,
        '파킨슨_OR': round(or_pd, 2),
        '파킨슨_p-value': round(p_pd, 4),
        '알츠하이머_OR': round(or_ad, 2),
        '알츠하이머_p-value': round(p_ad, 4)
    })

# 3. [최종 테이블 생성] 기존 final_report(환자수, 발병률 포함)와 병합
stats_df = pd.DataFrame(stats_results)
full_final_report = pd.merge(final_report, stats_df, on='수면세부코드')
#%%

# 오즈비 결과표 수정중

def get_priority(row):
    pd_or = row['파킨슨_OR']
    ad_or = row['알츠하이머_OR']
    pd_p  = row['파킨슨_p-value']
    ad_p  = row['알츠하이머_p-value']

    # 유의한 OR 분리 (위험 증가 vs 보호 효과)
    risk_ors    = []  # OR > 1 이고 p < 0.05 → 위험 증가
    protect_ors = []  # OR < 1 이고 p < 0.05 → 보호 효과

    if pd_p < 0.05:
        (risk_ors if pd_or >= 1 else protect_ors).append(pd_or)
    if ad_p < 0.05:
        (risk_ors if ad_or >= 1 else protect_ors).append(ad_or)

    # 위험 증가 등급 (OR > 1, p < 0.05)
    if risk_ors:
        max_or = max(risk_ors)
        if max_or >= 3.0:
            return "A등급 ★★★ (고위험/집중관리)"
        elif max_or >= 2.0:
            return "B등급 ★★ (중위험/주의관찰)"
        elif max_or >= 1.2:
            return "C등급 ★ (경도위험/일반관리)"
        else:
            return "C등급 ★ (경도위험/일반관리)"

    # 보호효과 → 상대적 저위험으로 변경
    if protect_ors:
        min_or = min(protect_ors)
        if min_or < 0.5:
            return "L1등급 🔵 (상대적 저위험 - 다른 수면장애 대비 발병률 낮음)"
        else:
            return "L2등급 🔵 (상대적 저위험 - 경미한 수준)"

    # 둘 다 없으면
    return "D등급 (통계 미유의)"

full_final_report['관리등급'] = full_final_report.apply(get_priority, axis=1)

# 5. 결과 출력
print("\n" + "="*50)
print("  [종합 분석 결과: 수면 장애별 뇌질환 이행 위험도 및 통계 검정]")
print("="*50)
display(full_final_report[[
    '수면세부코드', '수면장애_총환자수',
    '파킨슨_발병률(%)', '파킨슨_OR', '파킨슨_p-value',
    '알츠하이머_발병률(%)', '알츠하이머_OR', '알츠하이머_p-value',
    '평균진단연령', '관리등급'
]].sort_values('파킨슨_OR', ascending=False))


#%%
#질환별 상대적 위험도 비교 컬럼 추가 코드
# 1. 파킨슨 vs 알츠하이머 위험도 비교 로직 함수
def compare_risks(row):
    pd_or = row['파킨슨_OR']
    ad_or = row['알츠하이머_OR']

    # 두 OR 값이 모두 유효한지 확인 (NaN 제외)
    if pd.isna(pd_or) or pd.isna(ad_or):
        return "데이터 부족"

    # 오즈비(OR)가 더 높은 쪽을 선택
    if pd_or > ad_or:
        diff = pd_or - ad_or
        return f"파킨슨 상대적 높음 (+{diff:.2f})"
    elif ad_or > pd_or:
        diff = ad_or - pd_or
        return f"알츠하이머 상대적 높음 (+{diff:.2f})"
    else:
        return "위험도 동일"

# 2. 컬럼 적용
full_final_report['상대적_위험방향'] = full_final_report.apply(compare_risks, axis=1)

# 3. 결과 출력 (가독성을 위해 정렬)
print("\n" + "="*50)
print("  [수면 장애별 파킨슨 vs 알츠하이머 상대적 위험도 비교]")
print("="*50)

display(full_final_report[[
    '수면세부코드',
    '파킨슨_OR', '알츠하이머_OR',
    '상대적_위험방향', '관리등급'
]].sort_values('파킨슨_OR', ascending=False))



#%%
#수정
def assign_management_grade(row):
    pd_or  = row['파킨슨_OR']
    ad_or  = row['알츠하이머_OR']
    pd_p   = row['파킨슨_p']     # p-value 컬럼 필요
    ad_p   = row['알츠하이머_p'] # p-value 컬럼 필요

    if pd.isna(pd_or) or pd.isna(ad_or):
        return "데이터 부족"

    # 통계적으로 유의한 OR만 선별 (p < 0.05)
    valid_ors = []
    if pd_p < 0.05:
        valid_ors.append(pd_or)
    if ad_p < 0.05:
        valid_ors.append(ad_or)

    # 유의한 OR이 없으면 등급 보류
    if not valid_ors:
        return "D (통계 미유의)"

    max_or = max(valid_ors)

    # 유의한 OR 기준으로 등급 분류
    if max_or >= 3.0:
        return "A (고위험)"
    elif max_or >= 2.0:
        return "B (중위험)"
    elif max_or >= 1.5:
        return "C (저위험)"
    else:
        return "D (통계 미유의)"


#%%
"""### 처방일수가 1일 증가할수록 뇌질환에 걸릴 오즈(Odds)가 몇 배 증가하는가"""
#-------------------------------------------------------------------------------------
# 2. 타겟 변수 생성 (알츠하이머 또는 파킨슨 여부)
# 알츠하이머(G30), 파킨슨(G20) 코드가 포함되어 있으면 1, 아니면 0
df['target'] = df['주상병코드'].apply(lambda x: 1 if x in ['G30', 'G20'] else 0)

# 3. 분석 모델 구축 (로지스틱 회귀)
# 독립변수: 총처방일수 (여기에 성별, 연령대 등 통제변수를 추가하는 것이 정확합니다)
X = df[['총처방일수']]
X = sm.add_constant(X) # 상수항 추가
y = df['target']

model = sm.Logit(y, X).fit()

# 4. 결과 출력
print(model.summary())

# 5. 해석을 위한 오즈비(Odds Ratio) 계산
# 처방일수가 1일 증가할 때 질병 확률이 몇 배 증가하는지 나타냄
odds_ratios = np.exp(model.params)
print("\n[오즈비(Odds Ratio) 결과]")
print(odds_ratios)
#%%
#-------------------------------------------------------------------------------------

#  총처방일수 그룹을 상위, 중위, 하위로 나누기.

#  총처방일수의 인포를 확인 후 나누는 방법을 그에 맞춰서 분할하기위해 인포확인 먼저함
#  1. 총처방일수 인포
df['총처방일수'].hist(bins=50)
plt.title('Prescription Days Distribution')
plt.xlabel('Days')
plt.ylabel('Count')
plt.show()
# -----------------------------------------------------------------------------------------------------------
# 2.총처방일수 그룹을 상위, 중위, 하위로 나누기.
# 1. 그룹 나누기 (사용자 정의 기준)
bins = [-1, 3, 14, 960] # 범위 설정 (-1은 0을 포함하기 위함)
labels = ['하위(0-3일)', '중위(4-14일)', '상위(15일+)']

df['처방그룹'] = pd.cut(df['총처방일수'], bins=bins, labels=labels)

# 2. 그룹별 데이터 개수 확인
print(df['처방그룹'].value_counts().sort_index())

# 3. 알츠하이머/파킨슨 환자 비율 확인 (예시)
# 주상병코드에 G30(알츠하이머), G20(파킨슨)이 포함된 경우
df['target'] = df['주상병코드'].str.startswith(('G30', 'G20'), na=False).astype(int)

# 4. 그룹별 발병률 비교
pivot = df.groupby('처방그룹')['target'].mean() * 100 # 백분율(%)
print("\n[그룹별 알츠하이머/파킨슨 진료 비율]")
print(pivot)
#%%
"""## 5-3. 로지스틱회귀분석"""
#-------------------------------------------------------------------------------------
# alpaca_analysis_master = pd.read_csv("/content/drive/MyDrive/MedicalDA05/data/alpaca_analysis_master.csv", encoding="CP949")

#%%
# 연령대 범주 통합 

# '05-09세', '10-14세' 등을 하나하나 넣었더니 결과가 통계적으로 유의하지 않다고 나옴
# 현재 2만 건의 데이터 중 파킨슨 환자가 매우 적은데, '연령대' 변수가 너무 잘게 쪼개져 있는 것이 가장 큰 원인
# 따라서 '40세 미만'은 하나의 그룹으로 합치고, 그 이후부터 10살 단위로 묶을 필요가 있다
# 연령대_통합: 40세 미만 / 40대 / 50대 / 60대 / 70대 / 80세 이상

def regroup_age(age_str):
    # '00-04세' 같은 형식에서 앞의 숫자 두 자리만 추출
    try:
        age_num = int(age_str[:2])
    except:
        return '기타'

    if age_num < 40:
        return '40세미만'
    elif age_num < 50:
        return '40대'
    elif age_num < 60:
        return '50대'
    elif age_num < 70:
        return '60대'
    elif age_num < 80:
        return '70대'
    else:
        return '80세이상'

# 새로운 연령대 컬럼 생성
df_table1['연령대_통합'] = df_table1['연령대'].apply(regroup_age)

# 데이터가 잘 묶였는지 확인 (질환자 분포 확인)
print(df_table1.groupby('연령대_통합')['is_par_alz'].sum())
#%%
# """### Y(파킨슨) ~ X1(수면세부코드) + X2(연령대) + X3(성별)"""
# #-------------------------------------------------------------------------------------
# import statsmodels.api as sm

# # 1. 분석 데이터 준비 (연령대만 통합된 버전 사용)
# cols = ['수면세부코드', '연령대_통합', '성별', 'is_parkinson']
# df_mod_pk = df_table1[cols].dropna()

# # 2. 인코딩 (수면세부코드 전체 유지)
# X_pk = pd.get_dummies(df_mod_pk[['수면세부코드', '연령대_통합', '성별']], drop_first=True, dtype=float)
# X_pk = sm.add_constant(X_pk)
# y_pk = df_mod_pk['is_parkinson'].astype(float)

# # 3. 모델 학습 (L1 정규화 적용)
# try:
#     # alpha 값을 1.0에서 시작해보고, 여전히 에러가 나면 2.0, 5.0 식으로 높여보세요.
#     # L1 정규화는 불필요하거나 계산 불가능한 변수의 계수를 0으로 만들어 오류를 방지합니다.
#     model_pk = sm.Logit(y_pk, X_pk).fit_regularized(method='l1', alpha=2.0)
#     # fit_regularized: 통계 모델이 데이터의 노이즈나 '특정 상황(예: 환자 수가 0명인 경우)' 때문에 수치가 폭발하는 것을 막아준다

#     print("\n[최종 오즈비 결과]")
#     print(model_pk.summary())

#     # 오즈비 계산
#     pk_or = np.exp(model_pk.params)
#     print("\n[Odds Ratio]")
#     display(pk_or.sort_values(ascending=False))

# except Exception as e:
#     print(f"오류 발생: {e}")


# #%%


# #%%
# """### Y(알츠하이머) ~ X1(수면세부코드) + X2(연령대) + X3(성별)
# """
# #------------------------------------------------------------------------------------

# import statsmodels.api as sm

# # 1. 분석 데이터 준비 (연령대만 통합된 버전 사용)
# cols = ['수면세부코드', '연령대_통합', '성별', 'is_alzheimer']
# df_mod_pk = df_table1[cols].dropna()

# # 2. 인코딩 (수면세부코드 전체 유지)
# X_pk = pd.get_dummies(df_mod_pk[['수면세부코드', '연령대_통합', '성별']], drop_first=True, dtype=float)
# X_pk = sm.add_constant(X_pk)
# y_pk = df_mod_pk['is_alzheimer'].astype(float)

# # 3. 모델 학습 (L1 정규화 적용)
# try:
#     # alpha 값을 1.0에서 시작해보고, 여전히 에러가 나면 2.0, 5.0 식으로 높여보세요.
#     # L1 정규화는 불필요하거나 계산 불가능한 변수의 계수를 0으로 만들어 오류를 방지합니다.
#     model_pk = sm.Logit(y_pk, X_pk).fit_regularized(method='l1', alpha=2.0)
#     # fit_regularized: 통계 모델이 데이터의 노이즈나 '특정 상황(예: 환자 수가 0명인 경우)' 때문에 수치가 폭발하는 것을 막아준다

#     print("\n[알츠하이머병 위험도 분석 결과]")
#     print(model_pk.summary())

#     # 오즈비 계산
#     pk_or = np.exp(model_pk.params)
#     print("\n[최종 오즈비 결과]")
#     display(pk_or.sort_values(ascending=False))

# except Exception as e:
#     print(f"오류 발생: {e}")



# #%%
# """### Y(신경 퇴행성 뇌질환) ~ X1(수면세부코드) + X2(연령대) + X3(성별)
# """
# #-------------------------------------------------------------------------------------

# import pandas as pd
# import statsmodels.api as sm
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. 분석 데이터 준비
# # 연령대 통합과 is_par_alz 변수가 생성된 상태여야 합니다.
# cols = ['수면세부코드', '연령대_통합', '성별', 'is_par_alz']
# df_mod = df_table1[cols].dropna()

# # 2. 인코딩 (수면세부코드 전체 유지)
# X = pd.get_dummies(df_mod[['수면세부코드', '연령대_통합', '성별']], drop_first=True, dtype=float)
# X = sm.add_constant(X)
# y = df_mod['is_par_alz'].astype(float)

# # 3. 정규화 로지스틱 회귀 모델 학습 (L1 정규화)
# try:
#     # alpha=2.0은 정규화 강도입니다. 수치가 튄다면 이 값을 5.0까지 높여보세요.
#     model_combined = sm.Logit(y, X).fit_regularized(method='l1', alpha=2.0)

#     print("\n[신경 퇴행성 질환 통합 위험도 분석 결과]")
#     print(model_combined.summary())

#     # 4. 결과 정리 (오즈비 계산)
#     params = model_combined.params
#     conf = model_combined.conf_int()

#     results = pd.DataFrame({
#         'Odds Ratio': np.exp(params),
#         'P-value': model_combined.pvalues,
#         'Lower CI': np.exp(conf[0]),
#         'Upper CI': np.exp(conf[1])
#     })

#     # 5. 유의미한 결과만 보기 (P < 0.05)
#     print("\n[통계적으로 유의미한 변수 리스트]")
#     display(results[results['P-value'] < 0.05].sort_values('Odds Ratio', ascending=False))

# except Exception as e:
#     print(f"오류 발생: {e}")



#%%
"""### 장기처방 로지스틱 회귀분석
"""
#-------------------------------------------------------------------------------------
#  '15일 이상 장기 처방군'이 알츠하이머나 파킨슨병 진단과 얼마나 강력한 상관관계가 있는지 확인하는 분석코드

# 1. 분석에 필요한 컬럼만 추출 (메모리 절약)
# '주상병코드'와 '총처방일수'가 포함된 df를 사용합니다.
analysis_df = df[['총처방일수', '주상병코드']].copy()

# 2. 타겟 변수 생성 (알츠하이머 G30, 파킨슨 G20 계열)
# 해당 코드들로 시작하는 경우를 1(질환군), 아니면 0(대조군)으로 정의
analysis_df['is_disease'] = analysis_df['주상병코드'].str.startswith(('G30', 'G20', 'F00'), na=False).astype(int)

# 3. 처방일수 그룹화 (15일 기준 이진 분류 또는 3그룹 분류)
# 여기서는 '15일 이상 장기 처방 여부'가 질환에 미치는 영향을 직접 확인합니다.
analysis_df['is_long_term'] = (analysis_df['총처방일수'] >= 15).astype(int)

# 4. 로지스틱 회귀 분석 (통계적 유의성 검정)
# 독립변수(X): 15일 이상 처방 여부, 종속변수(y): 질환 여부
X = sm.add_constant(analysis_df['is_long_term'])
y = analysis_df['is_disease']

model = sm.Logit(y, X).fit()

# 5. 결과 해석 데이터 추출
summary = model.summary2().tables[1]
p_value = summary.loc['is_long_term', 'P>|z|']
odds_ratio = np.exp(summary.loc['is_long_term', 'Coef.'])

# 6. 결과 출력
print("--- 분석 결과 보고서 ---")
print(f"1. 15일 이상 처방군 비율: {(analysis_df['is_long_term'].mean()*100):.2f}%")
print(f"2. 전체 데이터 중 질환군 비율: {(analysis_df['is_disease'].mean()*100):.4f}%")
print(f"3. 오즈비(Odds Ratio): {odds_ratio:.4f}")
print(f"   (해석: 15일 이상 처방받은 경우, 그렇지 않은 경우보다 질환 진단 확률이 {odds_ratio:.2f}배 높음)")
print(f"4. P-value: {p_value:.10f}")

if p_value < 0.05:
  print("결과: 통계적으로 매우 유의미한 상관관계가 있음 (p < 0.05)")
else:
  print("결과: 통계적으로 유의미한 차이를 발견하지 못함")
#%%
"""### 65세 이상만 필터링"""
#-------------------------------------------------------------------------------------
import statsmodels.api as sm

# 1. 데이터 전처리 (기존 필터링 로직 유지)
# 65세 이상 & 수면장애(G47) 환자 추출
is_elderly = (df['연령대코드'] >= 14)
has_insomnia = (df['주상병코드'].str.contains('G47', na=False) | df['부상병코드'].str.contains('G47', na=False))
df_analysis = df[is_elderly & has_insomnia].copy()

# 중복 제거 (1인당 1건, 처방일수 많은 기록 우선)
df_analysis = df_analysis.sort_values(by='총처방일수', ascending=False)
df_analysis = df_analysis.drop_duplicates(subset='가입자일련번호', keep='first')

# 타겟 변수 생성 (질환 여부 0/1)
df_analysis['is_alzheimer'] = (df_analysis['주상병코드'].str.contains('G30', na=False) |
                               df_analysis['부상병코드'].str.contains('G30', na=False)).astype(int)
df_analysis['is_parkinson'] = (df_analysis['주상병코드'].str.contains('G20', na=False) |
                               df_analysis['부상병코드'].str.contains('G20', na=False)).astype(int)

print(f"✅ 분석 대상 환자 수: {len(df_analysis)}명")

# ---------------------------------------------------------
# 2. 로지스틱 회귀 분석 함수 (다변수 모델)
# ---------------------------------------------------------
def run_logit_analysis(target_col, disease_name):
    print(f"\n{'='*20} {disease_name} 분석 결과 {'='*20}")

    # 독립변수 설정: 총처방일수 + 연령대코드(통제변수)
    # 성별 코드가 있다면 여기에 추가하면 더 정확합니다.
    X = df_analysis[['총처방일수', '연령대코드']]
    X = sm.add_constant(X) # 상수항 추가
    y = df_analysis[target_col]

    try:
        model = sm.Logit(y, X).fit(disp=0)
        print(model.summary())

        # 오즈비(Odds Ratio) 및 95% 신뢰구간 계산
        params = model.params
        conf = model.conf_int()
        conf['OR'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        odds_ratios = np.exp(conf)

        print(f"\n[{disease_name} 오즈비(Odds Ratio) 해석]")
        print(odds_ratios.loc[['총처방일수', '연령대코드']])
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")

#%%
# ---------------------------------------------------------
# 3. 분석 실행
# ---------------------------------------------------------
run_logit_analysis('is_alzheimer', '알츠하이머(G30)')

#%%
run_logit_analysis('is_parkinson', '파킨슨(G20)')

#%%
"""### 처방일수 기준으로 그룹 나눔"""
#-------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# 1. [질환 여부 판별] 주/부상병코드에 G20(파킨슨), G30(알츠하이머)이 포함되는지 확인
# .str.contains를 사용하여 해당 코드가 포함된 환자를 True로 표시합니다.
df['is_parkinson'] = df['주상병코드'].str.contains('G20', na=False) | df['부상병코드'].str.contains('G20', na=False)
df['is_alzheimer'] = df['주상병코드'].str.contains('G30', na=False) | df['부상병코드'].str.contains('G30', na=False)

# 2. [그룹화] '총처방일수'를 3분위수(상/중/하)로 나눕니다.
try:
    df['처방일수_그룹'] = pd.qcut(df['총처방일수'], q=3, labels=['하(경증)', '중(중등증)', '상(중증)'])
except ValueError:
    # 데이터 중복이 많을 경우 처리
    df['처방일수_그룹'] = pd.qcut(df['총처방일수'], q=3, labels=['하(경증)', '중(중등증)', '상(중증)'], duplicates='drop')

def analyze_severity_risk(df, group_name, target_col):
    """
    기준군(하) 대비 특정 그룹(중/상)의 오즈비와 p-value 계산
    """
    is_target_group = (df['처방일수_그룹'] == group_name)
    is_reference_group = (df['처방일수_그룹'] == '하(경증)') # 기준점
    has_disease = (df[target_col] == True)

    # 2x2 테이블 (a:대상군O/질환O, b:대상군O/질환X, c:기준군O/질환O, d:기준군O/질환X)
    a = len(df[is_target_group & has_disease])
    b = len(df[is_target_group & ~has_disease])
    c = len(df[is_reference_group & has_disease])
    d = len(df[is_reference_group & ~has_disease])

    # Fisher's Exact Test
    try:
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
    except:
        odds_ratio, p_value = np.nan, np.nan

    return a, odds_ratio, p_value

# 3. [통계 계산 루프]
results = []
for g in ['중(중등증)', '상(중증)']:
    # 파킨슨(G20) 결과
    cnt_pd, or_pd, p_pd = analyze_severity_risk(df, g, 'is_parkinson')
    # 알츠하이머(G30) 결과
    cnt_ad, or_ad, p_ad = analyze_severity_risk(df, g, 'is_alzheimer')

    results.append({
        '처방강도(그룹)': g,
        '파킨슨_환자수': cnt_pd,
        '파킨슨_OR': round(or_pd, 2),
        '파킨슨_p': round(p_pd, 4),
        '알츠하이머_환자수': cnt_ad,
        '알츠하이머_OR': round(or_ad, 2),
        '알츠하이머_p': round(p_ad, 4)
    })

# 4. 결과 출력
report_df = pd.DataFrame(results)
print("\n" + "="*65)
print(" [총처방일수 심각도별 G20/G30 위험도 분석 결과 (기준: 하 그룹)]")
print("="*65)
display(report_df)
#%%
"""
==============================================================
  머신러닝
==============================================================
"""
#-------------------------------------------------------------------------------------
sleep_any_unique
#%%
"""## 연령대 재그룹"""

# 1. 매핑 딕셔너리 생성
# 기존 코드 1~18을 신규 그룹 1~6으로 연결해서 연령대 재그룹
age_mapping = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,  # 40세 미만
    9: 2, 10: 2,                                   # 40대
    11: 3, 12: 3,                                  # 50대
    13: 4, 14: 4,                                  # 60대
    15: 5, 16: 5,                                  # 70대
    17: 6, 18: 6                                   # 80세 이상
}

# 2. 새로운 컬럼 생성 
sleep_any_unique['연령대_재그룹'] = sleep_any_unique['연령대코드'].map(age_mapping)

# 4. 결과 확인
print(sleep_any_unique[['연령대코드', '연령대_재그룹']].head(10))
print(sleep_any_unique['연령대_재그룹'].value_counts().sort_index())
#%%
"""## 성별코드 정리"""
#-------------------------------------------------------------------------------------

sleep_any_unique['성별_재그룹'] = sleep_any_unique['성별코드'].map({1: 0, 2: 1})

#%%
"""## 수면세부코드 정리"""
#-------------------------------------------------------------------------------------
# 1. 수면장애 세부 분류 매핑
sleep_map = {     'G470': 'G470 불면증',     'G471': 'G471 과다수면',     'G472': 'G472 수면각성일정장애',     'G473': 'G473 수면무호흡',     'G474': 'G474 기면증',     'G478': 'G478 기타수면장애',     'G479': 'G479 상세불명수면장애' }


def get_sleep_map(code):
    if pd.isna(code): return '해당없음'
    clean_code = str(code).replace('.', '')[:4]
    return sleep_map.get(clean_code, '기타수면장애')

# 2. 세부명 추출 (주상병 -> 부상병 순)
sleep_any_unique['수면_세부명'] = sleep_any_unique['주상병코드'].apply(get_sleep_map)

# 주상병이 수면장애가 아닐 때만 부상병 확인
mask = (sleep_any_unique['수면_세부명'] == '기타수면장애') & (sleep_any_unique['수면장애_진단'] == 1)
sleep_any_unique.loc[mask, '수면_세부명'] = sleep_any_unique.loc[mask, '부상병코드'].apply(get_sleep_map)

# 3. 머신러닝을 위한 원-핫 인코딩 (컬럼 생성)
# '기타/비수면장애'와 '해당없음'을 제외한 실제 수면 질환들만 컬럼으로 변환
sleep_dummies = pd.get_dummies(sleep_any_unique['수면_세부명'])

# 분석에 유의미한 수면 질환 컬럼만 선택하여 기존 데이터에 결합
# (딕셔너리에 정의된 실제 질환명들만 추출)
valid_sleep_cols = [col for col in sleep_dummies.columns if col in sleep_map.values()]
sleep_any_unique = pd.concat([sleep_any_unique, sleep_dummies[valid_sleep_cols].astype(int)], axis=1)

# 결과 확인
print("### 생성된 수면 세부 컬럼 확인 ###")
print(sleep_any_unique[valid_sleep_cols].head())
#%%
# 시각화 설정
plt.figure(figsize=(12, 6))
sns.set_palette("viridis")

# '기타/비수면장애' 및 '해당없음' 제외하고 실제 수면 장애 환자만 필터링하여 카운트
plot_data = sleep_any_unique[sleep_any_unique['수면_세부명'].isin(sleep_map.values())]
order = plot_data['수면_세부명'].value_counts().index

ax = sns.countplot(data=plot_data, y='수면_세부명', order=order)

# 그래프 디테일
plt.title('수면장애 세부 질환별 환자 분포 (건수)', fontsize=15, pad=20)
plt.xlabel('환자 수(진료 건수)')
plt.ylabel('수면 질환 분류')

# 숫자 라벨링
for p in ax.patches:
    ax.annotate(f'{int(p.get_width()):,}건',
                (p.get_width(), p.get_y() + p.get_height()/2),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points')

plt.tight_layout()
plt.show()
#%%
# """#머신러닝=================================================================

# ##머신러닝 비교 최적 모델
# * Random Forest
# * XGBoost
# * LightGBM
# """
# #--------------------------------------------------------------------------------------
# # 터미널에서 xgboost 라이브러리 설치
# # conda install -c conda-forge xgboost

# # 터미널에서 lightgbm 라이브러리 설치
# # conda install -c conda-forge lightgbm
# #------------------------------------------------------------------------------
# #모델 비교 

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # --- [1] 성별 및 필수 변수 재확인 (에러 방지) ---
# # 성별코드가 1, 2로 되어 있다면 0, 1로 변환하여 '성별_01' 컬럼 생성
# if '성별코드' in sleep_any_unique.columns:
#     sleep_any_unique['성별_01'] = sleep_any_unique['성별코드'].apply(lambda x: 0 if x == 1 else 1)
# elif '성별_남성여부' in sleep_any_unique.columns:
#     # 이미 '성별_남성여부'가 있다면 그대로 사용하도록 매핑
#     sleep_any_unique['성별_01'] = sleep_any_unique['성별_남성여부']

# # 총처방일수 결측치 처리 및 수치형 변환
# sleep_any_unique['총처방일수'] = pd.to_numeric(sleep_any_unique['총처방일수'], errors='coerce').fillna(0)

# # --- [2] 학습 변수(Features) 설정 ---
# # 원-핫 인코딩으로 생성된 수면 세부 질환 컬럼들
# sleep_map_cols = {'G470':'불면증', 'G471':'과다수면', 'G472':'수면각성일정장애', 'G473':'수면무호흡', 'G474':'기면증', 'G478':'기타수면장애', 'G479':'상세불명수면장애'}
# existing_sleep_cols = [col for col in sleep_map_cols if col in sleep_any_unique.columns]

# # 기저질환 및 기타 변수 중 데이터프레임에 실제 존재하는 것만 선택
# base_features = ['성별_01', '연령대_재그룹', '총처방일수', '고혈압_진단', '지질대사_진단',
#                  '당뇨병_진단', '기관지염_진단', '무릎관절증_진단', '등통증_진단']
# existing_base_features = [col for col in base_features if col in sleep_any_unique.columns]

# # 최종 독립변수 리스트
# final_features = existing_base_features + existing_sleep_cols

# # --- [3] 데이터 준비 및 분할 ---
# X = sleep_any_unique[final_features].astype(float)
# y = sleep_any_unique['Par_Alz_진단'].astype(int)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # --- [4] 모델 비교 실행 ---
# models = {
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'),
#     "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
# }

# model_results = []

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     model_results.append({
#         "Model": name,
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Precision": precision_score(y_test, y_pred),
#         "Recall": recall_score(y_test, y_pred),
#         "F1-Score": f1_score(y_test, y_pred),
#         "AUC": roc_auc_score(y_test, y_prob)
#     })

# # 결과 출력
# result_df = pd.DataFrame(model_results).set_index("Model")
# print("### 모델 성능 비교 결과 ###")
# print(result_df)

# # 가장 좋은 모델 선정
# best_model_name = result_df['AUC'].idxmax()
# print(f"\n최적 모델: {best_model_name}")

# # 시각화
# result_df.plot(kind='bar', figsize=(10, 6))
# plt.title('Model Performance Comparison')
# plt.legend(loc='lower right')
# plt.show()
# #%%
# """1. 왜 XGBoost가 최적 모델인가요? (AUC 기준)

# 결과표를 보면 Accuracy(정확도)는 Random Forest가 가장 높지만, 최적 모델은 AUC가 가장 높은 XGBoost로 선정. 그 이유는...

# AUC (0.8702): 모델의 전반적인 '분류 능력'을 나타내는 지표입니다. 1에 가까울수록 모델이 환자(1)와 일반인(0)을 잘 구분한다는 뜻입니다. 세 모델 중 XGBoost가 질환 유무를 판별하는 잠재력이 가장 뛰어납니다.

# 데이터 불균형 (Log 정보): 로그를 보면 Number of positive: 118, negative: 16730으로 나옵니다. 전체 데이터 중 질환 환자가 **약 0.7%**뿐인 매우 극심한 불균형 데이터입니다.

# 이런 경우, 모든 사람을 "정상"이라고만 답해도 정확도는 99.3%가 나옵니다. 따라서 정확도(Accuracy)는 아무런 의미가 없으며, AUC가 모델의 실력을 보여주는 진짜 지표가 됩니다.


# ---


# 2. 성능 지표 상세 해석

# F1-Score (0.1212): 현재 점수가 매우 낮습니다. 이는 모델이 '정상'은 잘 맞추지만, 실제 '환자'를 찾아내는 정밀도와 재현율이 아직 낮다는 뜻입니다. 118명뿐인 환자 데이터를 학습하기엔 정보가 부족했거나, 추가적인 튜닝(과적합 방지 등)이 필요함을 시사합니다.

# LightGBM 경고 (Found whitespace): 컬럼명에 공백이 있어 언더바(_)로 바꿨다는 뜻입니다. 성능에는 지장이 없으니 안심하셔도 됩니다.

# BoostFromScore (pavg=0.007): 모델이 학습을 시작할 때 "아, 이 데이터는 환자가 0.7%밖에 없구나"라는 것을 인지하고 그 확률부터 예측을 시작했다는 로그입니다.


# ---


# 3. 결과 요약 및 결론
# 현재 데이터에서 XGBoost는 다른 모델들보다 **"비록 맞춘 환자 수는 적을지라도, 환자와 정상인을 구분하는 로직 자체는 가장 정교하게 학습했다"**고 평가할 수 있습니다.


# ---



# 모델 성능을 더 올리기 위한 제안
# 현재 F1-Score가 낮은 문제를 해결하여 실제 환자를 더 잘 찾게 만드는 방법들입니다:
# 데이터 불균형 처리 (SMOTE 등): 환자(118명) 데이터를 인위적으로 늘리거나, 학습 시 환자 데이터에 가중치를 더 주는 scale_pos_weight 파라미터를 사용하면 F1-Score가 크게 올라갈 수 있습니다.
# 변수 중요도 확인: 최적 모델로 뽑힌 XGBoost가 어떤 변수(예: 불면증, 연령 등)를 중요하게 생각했는지 확인하는 것이 다음 단계입니다.
# """
# #%%
# # 1. 최적 모델(XGBoost)의 변수 중요도 추출
# # 이전 단계에서 학습된 'models["XGBoost"]' 객체를 사용합니다.
# xgb_model = models["XGBoost"]
# importances = xgb_model.feature_importances_
# feature_names = X.columns

# # 2. 중요도 순으로 정렬하여 데이터프레임 생성
# feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_df = feature_df.sort_values(by='Importance', ascending=False)

# # 3. 시각화
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_df, palette='magma')

# plt.title('XGBoost 변수 중요도 (알츠하이머/파킨슨 예측 기여도)', fontsize=15, pad=20)
# plt.xlabel('Importance (중요도 점수)', fontsize=12)
# plt.ylabel('분석 변수', fontsize=12)
# plt.grid(axis='x', linestyle='--', alpha=0.6)

# # 수치 라벨링 (막대 끝에 퍼센트 표시)
# for i, v in enumerate(feature_df['Importance']):
#     plt.text(v, i, f' {v:.3f}', va='center', fontsize=10)

# plt.tight_layout()
# plt.show()

# # 4. 주요 결과 요약 출력
# print("### 상위 5개 핵심 변수 ###")
# print(feature_df.head(5))
# #%%
# """## SMOTE 적용

# ### XGBoost
# """
# #--------------------------------------------------------------------------------------
# # !pip install imbalanced-learn  # 설치가 안 되어 있다면 실행하세요.
# from imblearn.over_sampling import SMOTE
# from collections import Counter

# # 1. SMOTE 적용 전 데이터 분포 확인
# print(f"오버샘플링 전 타겟 분포: {Counter(y_train)}")

# # 2. SMOTE 설정 및 적용
# # sampling_strategy='auto'는 소수 집단을 다수 집단과 1:1 비율이 될 때까지 생성합니다.
# smote = SMOTE(random_state=42)
# X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

# print(f"오버샘플링 후 타겟 분포: {Counter(y_train_over)}")

# # 3. SMOTE된 데이터로 XGBoost 재학습
# # 환자 데이터가 충분해졌으므로 scale_pos_weight는 따로 설정하지 않아도 됩니다.
# xgb_smote = XGBClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     random_state=42,
#     eval_metric='logloss'
# )

# xgb_smote.fit(X_train_over, y_train_over)

# # 4. 성능 평가 (원본 테스트 데이터 X_test로 평가해야 함)
# y_pred_smote = xgb_smote.predict(X_test)
# y_prob_smote = xgb_smote.predict_proba(X_test)[:, 1]

# print("\n" + "="*30)
# print(f"SMOTE 적용 후 Accuracy: {accuracy_score(y_test, y_pred_smote):.4f}")
# print(f"SMOTE 적용 후 F1-Score: {f1_score(y_test, y_pred_smote):.4f}")
# print(f"SMOTE 적용 후 AUC: {roc_auc_score(y_test, y_prob_smote):.4f}")
# print("="*30)

# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter

# # --- [1] 데이터 분포 변화 시각화 ---
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # SMOTE 전 분포
# train_dist_before = Counter(y_train)
# sns.barplot(x=list(train_dist_before.keys()), y=list(train_dist_before.values()), ax=axes[0], palette='pastel')
# axes[0].set_title(f'SMOTE 적용 전 (환자 수: {train_dist_before[1]}명)', fontsize=13)
# axes[0].set_xticklabels(['정상(0)', '환자(1)'])
# axes[0].set_ylabel('데이터 수')

# # SMOTE 후 분포
# train_dist_after = Counter(y_train_over)
# sns.barplot(x=list(train_dist_after.keys()), y=list(train_dist_after.values()), ax=axes[1], palette='coolwarm')
# axes[1].set_title(f'SMOTE 적용 후 (환자 수: {train_dist_after[1]}명)', fontsize=13)
# axes[1].set_xticklabels(['정상(0)', '환자(1)'])

# plt.suptitle('학습 데이터 클래스 불균형 해소 (SMOTE)', fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# # --- [2] 모델 성능 변화 지표 비교 ---
# # 기존 XGBoost 결과와 SMOTE 후 XGBoost 결과를 합쳐서 비교
# performance_comparison = pd.DataFrame({
#     'Metric': ['Accuracy', 'F1-Score', 'AUC'],
#     'Before_SMOTE': [accuracy_score(y_test, models["XGBoost"].predict(X_test)),
#                      f1_score(y_test, models["XGBoost"].predict(X_test)),
#                      roc_auc_score(y_test, models["XGBoost"].predict_proba(X_test)[:, 1])],
#     'After_SMOTE': [accuracy_score(y_test, y_pred_smote),
#                     f1_score(y_test, y_pred_smote),
#                     roc_auc_score(y_test, y_prob_smote)]
# })

# # 그래프 그리기 위해 변환
# perf_melted = performance_comparison
# #%%
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from collections import Counter

# # --- [1] 데이터 분포 변화 시각화 (경고 해결 버전) ---
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # SMOTE 전 분포
# train_dist_before = Counter(y_train)
# keys_b = list(train_dist_before.keys())
# vals_b = list(train_dist_before.values())
# sns.barplot(x=keys_b, y=vals_b, ax=axes[0], hue=keys_b, palette='pastel', legend=False)
# axes[0].set_title(f'SMOTE 적용 전 (환자 수: {train_dist_before[1]}명)', fontsize=13)
# axes[0].set_xticks([0, 1])  # FixedLocator 관련 경고 해결
# axes[0].set_xticklabels(['정상(0)', '환자(1)'])
# axes[0].set_ylabel('데이터 수')

# # SMOTE 후 분포
# train_dist_after = Counter(y_train_over)
# keys_a = list(train_dist_after.keys())
# vals_a = list(train_dist_after.values())
# sns.barplot(x=keys_a, y=vals_a, ax=axes[1], hue=keys_a, palette='coolwarm', legend=False)
# axes[1].set_title(f'SMOTE 적용 후 (환자 수: {train_dist_after[1]}명)', fontsize=13)
# axes[1].set_xticks([0, 1])  # FixedLocator 관련 경고 해결
# axes[1].set_xticklabels(['정상(0)', '환자(1)'])

# plt.suptitle('학습 데이터 클래스 불균형 해소 (SMOTE)', fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# # --- [2] 모델 성능 변화 지표 비교 ---
# # 이전 XGBoost(Before)와 SMOTE 적용 XGBoost(After) 결과 비교
# performance_comparison = pd.DataFrame({
#     'Metric': ['Accuracy', 'F1-Score', 'AUC'],
#     'Before_SMOTE': [
#         accuracy_score(y_test, models["XGBoost"].predict(X_test)),
#         f1_score(y_test, models["XGBoost"].predict(X_test)),
#         roc_auc_score(y_test, models["XGBoost"].predict_proba(X_test)[:, 1])
#     ],
#     'After_SMOTE': [
#         accuracy_score(y_test, y_pred_smote),
#         f1_score(y_test, y_pred_smote),
#         roc_auc_score(y_test, y_prob_smote)
#     ]
# })

# # 그래프용 데이터 변환
# perf_melted = performance_comparison.melt(id_vars='Metric', var_name='SMOTE_Status', value_name='Score')

# plt.figure(figsize=(10, 6))
# ax = sns.barplot(data=perf_melted, x='Metric', y='Score', hue='SMOTE_Status', palette='muted')

# # 막대 위에 수치 표시
# for p in ax.patches:
#     height = p.get_height()
#     if height > 0: # 0보다 큰 경우만 표시
#         ax.annotate(f'{height:.3f}',
#                     (p.get_x() + p.get_width() / 2., height),
#                     ha='center', va='center', xytext=(0, 9),
#                     textcoords='offset points', fontsize=10, fontweight='bold')

# plt.title('SMOTE 적용 전후 모델 성능 비교', fontsize=15)
# plt.ylim(0, 1.1)
# plt.legend(loc='upper right', title='데이터 상태')
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.show()
# #%%
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_auc_score

# # 1. 독립변수(X)와 종속변수(y) 설정
# # (기존에 정의한 feature 리스트를 그대로 사용합니다)
# X = sleep_any_unique[final_features].astype(float)
# y = sleep_any_unique['Par_Alz_진단'].astype(int)

# # 2. 데이터 분할 (8:2 비율)
# # test_size=0.2 : 20%를 테스트용으로, 나머지 80%를 학습용으로 설정
# # random_state=42 : 난수 시드를 고정하여 실험의 재현성 확보
# # stratify=y : 타겟(질환 유무)의 비율을 학습/테스트셋에 동일하게 유지 (불균형 데이터 필수!)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print(f"전체 데이터 수: {len(X)}")
# print(f"학습용 데이터(80%): {len(X_train)}건")
# print(f"테스트용 데이터(20%): {len(X_test)}건")

# # 3. 모델 정의 및 80% 데이터로 학습(fit)
# model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
# model.fit(X_train, y_train)

# # 4. 나머지 20% 데이터로 성능 테스트(predict)
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]

# # 5. 결과 리포트 출력
# print("\n[최종 모델 평가 리포트]")
# print(classification_report(y_test, y_pred))
# print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
# #%%
# """### 세 가지 모델 비교"""
# #--------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import (accuracy_score, f1_score, recall_score,
#                              precision_score, roc_auc_score, classification_report)

# # --- [1] 데이터 준비 및 분할 (기존 변수 설정 유지) ---
# X = sleep_any_unique[final_features].astype(float)
# y = sleep_any_unique['Par_Alz_진단'].astype(int)

# # 층화 추출(stratify)을 통해 분할 시 클래스 비율 유지
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # --- [2] SMOTE 적용 (훈련 데이터셋에만 적용) ---
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# print(f"SMOTE 적용 전 학습 데이터: {len(X_train)}")
# print(f"SMOTE 적용 후 학습 데이터: {len(X_train_res)}")

# # --- [3] 모델 정의 ---
# models = {
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'),
#     "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
# }

# model_results = []

# # --- [4] 모델 학습 및 성능 측정 ---
# for name, model in models.items():
#     model.fit(X_train_res, y_train_res)  # 오버샘플링된 데이터로 학습
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     # 지표 계산
#     model_results.append({
#         "Model": name,
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Recall": recall_score(y_test, y_pred),
#         "Precision": precision_score(y_test, y_pred),
#         "F1-Score": f1_score(y_test, y_pred),
#         "AUC": roc_auc_score(y_test, y_prob)
#     })

# # --- [5] 결과 표 출력 ---
# result_df = pd.DataFrame(model_results).set_index("Model")
# print("\n### SMOTE 적용 모델 성능 비교 결과 표 ###")
# display(result_df)

# # --- [6] 결과 시각화 (Accuracy, F1-Score, AUC 세 가지 비교) ---
# # 요청하신 이미지와 유사한 다중 막대 그래프 형태
# plot_cols = ['Accuracy', 'F1-Score', 'AUC']
# ax = result_df[plot_cols].plot(kind='bar', figsize=(12, 7), width=0.8)

# plt.title('Model Performance Comparison (SMOTE Applied)', fontsize=15, pad=20)
# plt.ylabel('Score', fontsize=12)
# plt.xlabel('Machine Learning Models', fontsize=12)
# plt.xticks(rotation=0)
# plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.ylim(0, 1.1)  # 여유 공간 확보

# # 막대 위에 수치 표시
# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.3f}',
#                 (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center',
#                 xytext=(0, 9),
#                 textcoords='offset points',
#                 fontsize=9)

# plt.tight_layout()
# plt.show()
# #%%
# """## 가중치 옵션 적용

# ### XGBoost
# """
# #--------------------------------------------------------------------------------------
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 1. 가중치 계산 (정상군 수 / 질환군 수)
# # y_train의 0과 1의 개수를 자동으로 계산하여 적용합니다.
# counts = y_train.value_counts()
# ratio = counts[0] / counts[1]
# print(f"계산된 가중치(scale_pos_weight): {ratio:.2f}")

# # 2. 모델 정의 (가중치 옵션 추가)
# # scale_pos_weight=ratio 를 통해 소수 클래스에 힘을 실어줍니다.
# weighted_model = XGBClassifier(
#     n_estimators=100,
#     random_state=42,
#     scale_pos_weight=ratio,
#     eval_metric='logloss'
# )

# # 3. 모델 학습
# weighted_model.fit(X_train, y_train)

# # 4. 예측 및 확률 계산
# y_pred_w = weighted_model.predict(X_test)
# y_prob_w = weighted_model.predict_proba(X_test)[:, 1]

# # 5. 결과 리포트 출력
# print("\n[가중치 적용 후 최종 모델 평가 리포트]")
# print(classification_report(y_test, y_pred_w))
# print(f"AUC Score: {roc_auc_score(y_test, y_prob_w):.4f}")

# # 6. 혼동 행렬 시각화 (환자를 몇 명 맞췄는지 시각적으로 확인)
# plt.figure(figsize=(6, 4))
# cm = confusion_matrix(y_test, y_pred_w)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
# plt.title('Confusion Matrix (Weighted XGBoost)')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
# #%%
# """### 세 가지 모델 비교"""
# #--------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import (accuracy_score, f1_score, recall_score,
#                              precision_score, roc_auc_score)

# # --- [1] 데이터 준비 및 분할 ---
# X = sleep_any_unique[final_features].astype(float)
# y = sleep_any_unique['Par_Alz_진단'].astype(int)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # --- [2] 가중치 비율 계산 ---
# # 대조군(0)과 실험군(1)의 비율을 계산하여 가중치로 설정 (약 45배)
# # 원래 데이터에는 대조군이 실험군의 약 45배였음
# # 하지만 현재 학습용 데이터(y_train)에는 대조군이 실험군의 약 142배여서 가중치가 더 극단적으로 적용됨
# ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])
# # ratio=45
# print(f"설정된 가중치 비율(Scale_pos_weight): {ratio:.2f}")

# # --- [3] 모델 정의 (가중치 옵션 적용) ---
# models = {
#     "Random Forest": RandomForestClassifier(
#         n_estimators=100,
#         class_weight='balanced', # RF는 'balanced' 키워드 사용
#         random_state=42
#     ),
#     "XGBoost": XGBClassifier(
#         n_estimators=100,
#         scale_pos_weight=ratio,  # XGBoost 가중치
#         learning_rate=0.1,
#         random_state=42,
#         eval_metric='logloss'
#     ),
#     "LightGBM": LGBMClassifier(
#         n_estimators=100,
#         scale_pos_weight=ratio,  # LightGBM 가중치
#         learning_rate=0.1,
#         random_state=42,
#         verbose=-1
#     )
# }

# model_results = []

# # --- [4] 모델 학습 및 성능 측정 ---
# for name, model in models.items():
#     model.fit(X_train, y_train)  # 가중치 옵션을 사용하므로 원본 데이터 그대로 학습
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     # 지표 계산
#     model_results.append({
#         "Model": name,
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Recall": recall_score(y_test, y_pred),
#         "Precision": precision_score(y_test, y_pred),
#         "F1-Score": f1_score(y_test, y_pred),
#         "AUC": roc_auc_score(y_test, y_prob)
#     })

# # --- [5] 결과 표 출력 ---
# result_df = pd.DataFrame(model_results).set_index("Model")
# print("\n### 가중치 적용 모델 성능 비교 결과 표 ###")
# display(result_df)

# # --- [6] 결과 시각화 (Accuracy, F1-Score, AUC 세 가지 비교) ---
# plot_cols = ['Accuracy', 'F1-Score', 'AUC']
# ax = result_df[plot_cols].plot(kind='bar', figsize=(12, 7), width=0.8)

# plt.title('Model Performance Comparison (Weighted Models)', fontsize=15, pad=20)
# plt.ylabel('Score', fontsize=12)
# plt.xlabel('Machine Learning Models', fontsize=12)
# plt.xticks(rotation=0)
# plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.ylim(0, 1.1)

# # 막대 위에 수치 표시
# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.3f}',
#                 (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center',
#                 xytext=(0, 9),
#                 textcoords='offset points',
#                 fontsize=9)

# plt.tight_layout()
# plt.show()
# #%%
# """## 데이터 불균형 해결: 언더샘플링 + easy앙상블


# """
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
# from imblearn.ensemble import EasyEnsembleClassifier
# from xgboost import XGBClassifier

# # ============================================================
# # 1. 데이터 로드 및 전처리
# # ============================================================

# # 각자 데이터 저장된 경로에 맞게 수정
# file_path = r'D:\project_semi1\국민건강보험공단_진료내역정보_2023.CSV'
# # file_path = r'E:\Semi1_ALPACARE\data\국민건강보험공단_진료내역정보_2023.CSV.CSV'

# print("데이터를 불러오는 중입니다...")
# df = pd.read_csv(file_path, encoding='cp949', usecols=['가입자일련번호', '성별코드', '연령대코드', '주상병코드', '부상병코드'])

# # NaN 처리 및 수면/뇌질환 플래그 생성
# df[['주상병코드', '부상병코드']] = df[['주상병코드', '부상병코드']].fillna('')
# 주상, 부상 = df['주상병코드'].astype(str), df['부상병코드'].astype(str)

# df['has_G47'] = (주상.str.startswith('G47') | 부상.str.startswith('G47')).astype(int)
# df['has_brain'] = (주상.str.contains('G20|G30') | 부상.str.contains('G20|G30')).astype(int)

# # 수면 세부 질환 매핑
# sleep_map = {     'G470': 'G470 불면증',     'G471': 'G471 과다수면',     'G472': 'G472 수면각성일정장애',     'G473': 'G473 수면무호흡',     'G474': 'G474 기면증',     'G478': 'G478 기타수면장애',     'G479': 'G479 상세불명수면장애' }
# for code, col in sleep_map.items():
#     df[col] = (주상.str.startswith(code) | 부상.str.startswith(code)).astype(int)

# # ============================================================
# # 2. 환자 단위 집계 및 타겟 정의
# # ============================================================
# agg_dict = {'성별코드':'first', '연령대코드':'first', 'has_G47':'max', 'has_brain':'max'}
# agg_dict.update({col: 'max' for col in sleep_map.values()})

# patient_df = df.groupby('가입자일련번호', sort=False).agg(agg_dict).reset_index()

# # 수면장애 환자만 추출 (질환군 정의)
# patient_df = patient_df[patient_df['has_G47'] == 1].copy()
# patient_df['Par_Alz_진단'] = patient_df['has_brain'].astype(int)

# # 연령대/성별 전처리
# age_mapping = {i: 1 if i <= 8 else (i-7)//2 + 1 for i in range(1, 19)} # 간소화된 매핑 logic
# patient_df['연령대_재그룹'] = patient_df['연령대코드'].map(age_mapping)
# patient_df['성별_01'] = patient_df['성별코드'].map({1: 0, 2: 1})
# patient_df = patient_df.dropna(subset=['연령대_재그룹', '성별_01'])

# # ============================================================
# # 3. 머신러닝 학습 (EasyEnsemble)
# # ============================================================
# features = ['성별_01', '연령대_재그룹'] + list(sleep_map.values())
# X = patient_df[features].astype(int)
# y = patient_df['Par_Alz_진단'].astype(int)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print(f"\n[데이터 현황] 전체:{len(patient_df):,} | 질환군:{y.sum():,} | 정상군:{(y==0).sum():,}")

# ee_model = EasyEnsembleClassifier(
#     n_estimators=10,
#     estimator=XGBClassifier(n_estimators=50, learning_rate=0.1, eval_metric='logloss'),
#     random_state=42, n_jobs=-1
# ).fit(X_train, y_train)

# # ============================================================
# # 4. 결과 출력 및 유덴 지수 최적화
# # ============================================================
# y_prob = ee_model.predict_proba(X_test)[:, 1]
# y_pred_init = ee_model.predict(X_test)

# # 유덴 지수 계산
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# j_scores = tpr + (1 - fpr) - 1
# best_idx = np.argmax(j_scores)
# best_threshold = thresholds[best_idx]

# # 결과 출력
# print("\n" + "="*60)
# print(f"### 1. 기본 분석 결과 (Threshold: 0.5) ###")
# print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
# print(classification_report(y_test, y_pred_init))

# print("\n" + "="*60)
# print(f"### 2. 유덴 지수 최적화 결과 (Threshold: {best_threshold:.4f}) ###")
# y_pred_youden = (y_prob >= best_threshold).astype(int)
# print(f"Max Youden's Index (J): {j_scores[best_idx]:.4f}")
# print(classification_report(y_test, y_pred_youden))
# print("="*60)

# # ============================================================
# # 5. 시각화 (한 번에 출력)
# # ============================================================
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False

# fig = plt.figure(figsize=(18, 5))

# # [1] 혼동 행렬 (최적화 후 기준)
# ax1 = fig.add_subplot(131)
# sns.heatmap(confusion_matrix(y_test, y_pred_youden), annot=True, fmt='d', cmap='Reds', ax=ax1)
# ax1.set_title('최적화 후 혼동 행렬')
# ax1.set_xlabel('예측값'); ax1.set_ylabel('실제값')

# # [2] 특성 중요도
# ax2 = fig.add_subplot(132)
# importances = np.mean([est.named_steps['classifier'].feature_importances_ for est in ee_model.estimators_], axis=0)
# feat_df = pd.DataFrame({'특성': features, '중요도': importances}).sort_values('중요도')
# ax2.barh(feat_df['특성'], feat_df['중요도'], color='#D85A30')
# ax2.set_title('특성 중요도 (Feature Importance)')

# # [3] ROC 커브 및 유덴 지수 포인트
# ax3 = fig.add_subplot(133)
# ax3.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}')
# ax3.plot([0, 1], [0, 1], 'k--')
# ax3.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, label=f'Best Threshold ({best_threshold:.2f})')
# ax3.set_title('ROC Curve & Youden\'s Index')
# ax3.set_xlabel('False Positive Rate'); ax3.set_ylabel('True Positive Rate')
# ax3.legend()

# plt.tight_layout()
# plt.show()


# %%


#%%
# ============================================================
# 수면장애 기반 뇌질환 예측 모델
# ============================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#%%

# 1. 변수 준비
sleep_map = {'G470':'불면증', 'G471':'과다수면', 'G472':'수면각성일정장애', 'G473':'수면무호흡', 'G474':'기면증', 'G478':'기타수면장애', 'G479':'상세불명수면장애'}
for code, name in sleep_map.items():
    sleep_any_unique[name] = (sleep_any_unique['주상병코드'].str.startswith(code, na=False) | 
                              sleep_any_unique['부상병코드'].str.startswith(code, na=False)).astype(int)

final_features = ['성별_재그룹', '연령대_재그룹'] + list(sleep_map.values())
X = sleep_any_unique[final_features].values  # 넘파이 배열로 변환하여 충돌 방지
y = sleep_any_unique['Par_Alz_진단'].values.astype(int)

# 2. 모델 설정
ratio = (len(y) - sum(y)) / sum(y)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, 
                             objective='binary:logistic', eval_metric='logloss', scale_pos_weight=ratio),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, class_weight='balanced', verbose=-1)
}

# 3. Stratified K-Fold (층화 k-폴드)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_results = []

for name, model in models.items():
    print(f"{name} 분석 중...")
    metrics = {'acc': [], 'pre': [], 'rec': [], 'f1': [], 'auc': []}
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 학습 및 예측
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # 여기서 확률값을 직접 추출
        
        # 지표 계산
        metrics['acc'].append(accuracy_score(y_test, y_pred))
        metrics['pre'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['rec'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics['auc'].append(roc_auc_score(y_test, y_prob))
    
    final_results.append({
        "Model": name,
        "Accuracy": np.mean(metrics['acc']),
        "Precision": np.mean(metrics['pre']),
        "Recall": np.mean(metrics['rec']),
        "F1-Score": np.mean(metrics['f1']),
        "AUC": np.mean(metrics['auc'])
    })

# 4. 결과 출력
result_df = pd.DataFrame(final_results).set_index("Model")
print("\n" + "="*60)
print("### [검증 완료] 최종 모델 성능 결과 ###")
print("="*60)
print(result_df.round(3))

# ============================================================
# 5. 시각화 개선 (x축 라벨 가로 정렬 및 디자인)
# ============================================================
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 막대 그래프 생성
ax = result_df[['Recall', 'AUC']].plot(
    kind='bar', 
    figsize=(10, 6), 
    width=0.7
)

# [핵심] x축 라벨을 가로로 정렬 (rotation=0)
plt.xticks(rotation=0, fontsize=11) 
plt.yticks(fontsize=10)

# 제목 및 라벨 설정
plt.title('최종 모델 성능 비교 (Recall vs AUC)', fontsize=15, pad=20)
plt.ylabel('Score (0.0 ~ 1.0)', fontsize=12)
plt.xlabel('Model Name', fontsize=12)

# 범례 위치 조정
plt.legend(loc='lower right', shadow=True)

# 수치 데이터 표시 (막대 위에 소수점 2자리까지)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 7), 
                textcoords = 'offset points',
                fontsize=10, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
#%%
"""# 데이터 불균형 해결: easy앙상블"""
# 필요한 라이브러리 설치 확인: pip install imbalanced-learn
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

#%%
# ============================================================
# [최종 완성] EasyEnsemble + 유덴 지수 최적화 + 3종 시각화
# ============================================================
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score

# 1. 데이터 준비 (수면 세부 질환 매핑 및 변수 설정)
sleep_map = {'G470':'불면증', 'G471':'과다수면', 'G472':'수면각성일정장애', 
             'G473':'수면무호흡', 'G474':'기면증', 'G478':'기타수면장애', 'G479':'상세불명수면장애'}

for code, name in sleep_map.items():
    sleep_any_unique[name] = (sleep_any_unique['주상병코드'].str.startswith(code, na=False) | 
                              sleep_any_unique['부상병코드'].str.startswith(code, na=False)).astype(int)

features = ['성별_재그룹', '연령대_재그룹'] + list(sleep_map.values())
X = sleep_any_unique[features].values
y = sleep_any_unique['Par_Alz_진단'].values.astype(int)

# 2. EasyEnsemble 모델 정의 (내부 분류기로 XGBoost 사용)
# n_estimators=10: 다수 범주를 10개로 나누어 10개의 모델을 앙상블함 (정보 손실 방지)
ee_model = EasyEnsembleClassifier(
    n_estimators=10,
    estimator=XGBClassifier(n_estimators=50, learning_rate=0.1, eval_metric='logloss', random_state=42),
    random_state=42, 
    n_jobs=1 # 충돌 방지
)

# 3. Stratified K-Fold 기반 학습 및 유덴 지수 추출
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 마지막 폴드 데이터를 시각화용으로 저장
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ee_model.fit(X_train, y_train)

# 4. 성능 평가 및 유덴 지수(Youden's Index) 최적화
y_prob = ee_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 유덴 지수 계산: J = Sensitivity + Specificity - 1
j_scores = tpr + (1 - fpr) - 1
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

# 최적 임계값 적용 예측
y_pred_youden = (y_prob >= best_threshold).astype(int)

print("\n" + "="*60)
print(f"### EasyEnsemble 모델 분석 결과 (최적 임계값: {best_threshold:.4f}) ###")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Max Youden's Index: {j_scores[best_idx]:.4f}")
print("-" * 60)
print(classification_report(y_test, y_pred_youden))
print("="*60)

# 5. 시각화 (3종 세트)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 5))

# [1] 혼동 행렬
ax1 = fig.add_subplot(131)
sns.heatmap(confusion_matrix(y_test, y_pred_youden), annot=True, fmt='d', cmap='Reds', ax=ax1)
ax1.set_title('최적화 후 혼동 행렬 (Confusion Matrix)')
ax1.set_xlabel('예측값'); ax1.set_ylabel('실제값')

# [2] 특성 중요도 (앙상블된 모델들의 평균치 계산)
ax2 = fig.add_subplot(132)
# EasyEnsemble 내부의 각 XGBoost 모델들의 중요도를 평균냄
importances = np.mean([est.named_steps['classifier'].feature_importances_ for est in ee_model.estimators_], axis=0)
feat_df = pd.DataFrame({'특성': features, '중요도': importances}).sort_values('중요도')
ax2.barh(feat_df['특성'], feat_df['중요도'], color='#D85A30')
ax2.set_title('특성 중요도 (Feature Importance)')

# [3] ROC 커브 및 최적 포인트
ax3 = fig.add_subplot(133)
ax3.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax3.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, label=f'최적 임계값 ({best_threshold:.2f})')
ax3.set_title('ROC Curve & Youden\'s Index')
ax3.set_xlabel('False Positive Rate'); ax3.set_ylabel('True Positive Rate')
ax3.legend()

plt.tight_layout()
plt.show()
# %%
"""#모델 성능 비교"""
# ============================================================
# [최종] 4종 모델 성능 통합 비교 (RF, XGB, LGBM, EasyEnsemble)
# ============================================================
from imblearn.ensemble import EasyEnsembleClassifier

# 1. 모델 리스트 정의
ratio = (len(y) - sum(y)) / sum(y)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, 
                             objective='binary:logistic', eval_metric='logloss', scale_pos_weight=ratio),
    "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, 
                               class_weight='balanced', verbose=-1),
    "EasyEnsemble": EasyEnsembleClassifier(n_estimators=10, 
                                           estimator=XGBClassifier(n_estimators=50, learning_rate=0.1, eval_metric='logloss'),
                                           random_state=42, n_jobs=1)
}

# 2. 통합 성능 평가 실행 (5-Fold 교차 검증)
all_model_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"{name} 모델 분석 중...")
    metrics = {'acc': [], 'pre': [], 'rec': [], 'f1': [], 'auc': []}
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics['acc'].append(accuracy_score(y_test, y_pred))
        metrics['pre'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['rec'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics['auc'].append(roc_auc_score(y_test, y_prob))
    
    all_model_results.append({
        "Model": name,
        "Accuracy": np.mean(metrics['acc']),
        "Precision": np.mean(metrics['pre']),
        "Recall": np.mean(metrics['rec']),
        "F1-Score": np.mean(metrics['f1']),
        "AUC": np.mean(metrics['auc'])
    })

# 3. 비교 데이터프레임 생성 및 출력
comparison_df = pd.DataFrame(all_model_results).set_index("Model")
print("\n" + "="*70)
print("### [전체 모델 성능 비교 리포트] ###")
print("="*70)
print(comparison_df.sort_values(by='AUC', ascending=False).round(3))

# 4. 통합 비교 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시각화할 지표 선택 (의료 데이터에서 가장 중요한 Recall과 AUC 중심)
ax = comparison_df[['Recall', 'AUC']].sort_values(by='AUC').plot(
    kind='bar', figsize=(12, 7), width=0.8
)

# 수치 표시 및 서식 설정
plt.xticks(rotation=0, fontsize=11)
plt.title('4종 모델 성능 통합 비교 (Recall & AUC)', fontsize=16, pad=20)
plt.ylabel('Score', fontsize=12)
plt.legend(loc='lower right', shadow=True)

# 막대 위에 수치 라벨 추가
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()





#%%