import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import time

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="Alpa-Ca", page_icon="🦙", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""

<style>
    /* 1. 기본 레이아웃 및 사이드바 숨기기 */
    [data-testid="collapsedControl"] { display: none; }
    .main { background-color: #ffffff; }

    /* 2. 체크박스 컨테이너 정렬 및 텍스트 설정 */
    div[data-testid="stCheckbox"] label {
        display: flex !important;
        align-items: center !important; 
        justify-content: flex-start !important;
        gap: 12px !important; 
        min-height: 40px !important;
    }

    /* 체크박스 텍스트 */
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] * {
        font-size: 24px !important; 
        font-weight: 400 !important; 
        color: #1e293b !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
        font-family: 'Pretendard', sans-serif !important;
    }

    /* 3. 체크박스 네모 칸(아이콘) 크기 조절 */
    div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] div div {
        transform: scale(1.2) !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }

    /* 체크박스 항목 간 세로 간격 */
    div[data-testid="stCheckbox"] {
        margin-bottom: 12px !important;
        padding: 5px 0 !important;
    }

    /* 4. 결과창 등급 배지 및 가이드 박스 */
    .grade-badge {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 30px;
        font-size: 1.5rem !important;
        font-weight: bold;
        margin: 10px 0;
    }
    .grade-safe    { background-color: #d4edda; color: #155724; }
    .grade-caution { background-color: #fff3cd; color: #856404; }
    .grade-warning { background-color: #ffe0b2; color: #e65100; }
    .grade-danger  { background-color: #f8d7da; color: #721c24; }

    .guide-box, .guide-box-urgent {
        padding: 20px 25px !important;
        border-radius: 12px !important;
        margin-bottom: 15px !important;
        font-size: 1.6rem !important;
    }
    .guide-box { background: #f0f7ff; border-left: 8px solid #2563eb; }
    .guide-box-urgent { background: #fff5f5; border-left: 8px solid #dc2626; }

    /* 5. 네비게이션 버튼 및 홈 화면 */
    [data-testid="stButton"] button[kind="primary"] {
        background-color: #2563eb;
        color: white;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        border-radius: 50px !important;
    }

    .stButton > button[kind="secondary"] p {
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }

    .hero-container h1 { font-size: 3.0rem !important; font-weight: 800; }
</style>

""", unsafe_allow_html=True)

# --- 페이지 라우팅(이동)을 위한 세션 상태 초기화 ---
if 'page' not in st.session_state:
    st.session_state.page = '홈'

def set_page(page_name):
    st.session_state.page = page_name

# --- 1. 모델 및 데이터 로드 ---
@st.cache_resource
def load_model():
    try:
        # 모델 패키지(딕셔너리)를 불러와 실제 모델과 임계값을 분리
        loaded_data = joblib.load('ensemble_sleep_model.pkl')
        real_model = loaded_data['model']
        best_thresh = loaded_data['best_threshold']
        return real_model, best_thresh
    except Exception as e:
        # 🚨 에러가 났을 때 웹 화면에 붉은색 에러 메시지로 원인을 알려줍니다.
        st.error(f"모델 파일을 불러오지 못했습니다. 에러 내용: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
       
        risk_table = pd.read_csv('risk_table.csv')
        overall = pd.read_csv('overall_table.csv', index_col=0)
        
        return {'risk_table': risk_table, 'overall': overall}

    except Exception as e:
        st.error(f"통계 데이터 로드 오류: {e}")
        return None

def get_age_code(birth_date):
    age = date.today().year - birth_date.year
    code = (age // 5) + 1
    return min(max(code, 1), 18), age


# --- 2. 수면 점수 / 등급 산출 ---
def calc_sleep_score(matched_codes, data, age_code, gender_code):
    if data is None or not matched_codes:
        return 0.0

    risk_table = data['risk_table']
    prob_safe = 1.0  # 모두 안전할 확률 (100%)

    for code in matched_codes:
        row = risk_table[
            (risk_table['연령대코드'] == age_code) &
            (risk_table['성별코드'] == gender_code) &
            (risk_table['수면세부코드'] == code)
        ]
        
        if row.empty or row['환자수'].values[0] < 5:
            row = risk_table[risk_table['수면세부코드'] == code]
            if row.empty:
                continue
            risk_percent = ((row['파킨슨'].sum() + row['알츠하이머'].sum()) / row['환자수'].sum() * 100)
        else:
            risk_percent = row['위험도'].values[0]

        prob_safe *= (1 - (risk_percent / 100.0))

    combined_risk = (1.0 - prob_safe) * 100
    return round(min(combined_risk, 100.0), 2)

# AI가 계산한 최적의 임계값을 반영하여 등급을 산출하도록 수정
def get_grade(score, threshold_percent):
    if score == 0 or score < (threshold_percent * 0.5):
        return "정상", "safe", "✅"
    elif score < threshold_percent:
        return "주의", "caution", "🟡"
    elif score < (threshold_percent * 1.5):
        return "경고", "warning", "🟠"
    else:
        return "위험", "danger", "🔴"

# 단계별 행동 가이드 데이터
GUIDES = {
    'G478': {
        'name': '기타 수면장애',
        'now': [
            "침대 옆 날카로운 물건 치우기 (낙상·부상 예방)",
            "파트너 또는 가족에게 증상 관찰 요청",
            "신경과 진료 예약 (파킨슨 전조 여부 확인 필수)",
        ],
        'long': [
            "6개월~1년 주기로 신경과 정기 추적 검사",
            "규칙적인 유산소 운동으로 뇌 신경 보호",
            "수면다원검사(PSG)로 렘수면 행동장애 공식 진단",
        ],
        'urgent': True
    },
    'G473': {
        'name': '수면무호흡',
        'now': [
            "옆으로 자는 자세 시도 (등을 대고 자면 기도 막힘 악화)",
            "음주·수면제 복용 자제",
            "이비인후과 또는 수면의학과 전문의 상담",
        ],
        'long': [
            "수면다원검사로 무호흡 중증도 확인",
            "양압기(CPAP) 사용 여부 전문의와 협의",
            "체중 조절 (비만은 무호흡 주요 원인)",
        ],
        'urgent': True
    },
    'G474': {
        'name': '발작수면 및 허탈발작',
        'now': [
            "운전·기계 조작 등 위험한 활동 자제",
            "짧은 낮잠(10~20분) 전략적으로 활용",
            "신경과 방문을 통한 뇌척수액 검사 상담",
        ],
        'long': [
            "규칙적인 수면 스케줄 유지",
            "카페인은 졸음 직전에 소량만 섭취",
            "처방약 필요 여부 전문의와 상의",
        ],
        'urgent': True
    },
    'G470': {
        'name': '불면증',
        'now': [
            "취침 1시간 전 스마트폰·TV 끄기",
            "침실 온도 18~20°C, 완전 차광 유지",
            "졸리지 않으면 억지로 눕지 않기",
        ],
        'long': [
            "매일 같은 시간에 기상 (주말 포함)",
            "낮에 30분 이상 햇볕 쬐기 (멜라토닌 리듬 회복)",
            "인지행동치료(CBT-I) 프로그램 참여 고려",
        ],
        'urgent': False
    },
    'G471': {
        'name': '과다수면',
        'now': [
            "낮잠은 오후 3시 이전, 20분 이내로 제한",
            "기상 즉시 커튼 열어 햇볕 쬐기",
            "고탄수화물 아침식사 피하기 (혈당 급등 방지)",
        ],
        'long': [
            "갑상선·빈혈 등 내과적 원인 검사",
            "규칙적인 유산소 운동으로 뇌 각성 시스템 강화",
            "수면다원검사로 수면의 질 점검",
        ],
        'urgent': False
    },
    'G472': {
        'name': '수면각성일정장애',
        'now': [
            "취침·기상 시간을 오늘부터 30분 단위로 점진적 조정",
            "저녁 이후 강한 조명·블루라이트 차단",
            "야식 금지 (식사 시간도 생체시계에 영향)",
        ],
        'long': [
            "광치료(Light Therapy) 기기 사용 고려",
            "멜라토닌 보충제 사용 전문의 상담 후 시도",
            "교대 근무자라면 근무 패턴 조정 요청",
        ],
        'urgent': False
    },
    'G479': {
        'name': '상세불명의 수면장애',
        'now': [
            "수면 일기 작성 시작 (잠든 시간, 깬 횟수, 아침 컨디션)",
            "카페인 섭취를 오후 2시 이후로는 중단",
        ],
        'long': [
            "2주 이상 지속되면 수면의학과 방문",
            "심리적 스트레스 및 불안 척도 평가 병행",
        ],
        'urgent': False
    },
}

# --- 3. 로드 ---
model, best_threshold = load_model()
final_df = load_data()


# =========================================================
# --- 4. 상단 네비게이션 바 (GNB) ---
# =========================================================
st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)

# 좌측에 로고, 우측에 탭 버튼 배치
nav_cols = st.columns([0.8, 0.8, 3, 1, 1, 1])

with nav_cols[0]:
    # 💡 [로고 1] 
    st.image("alpacalogo.png", use_container_width=True)

with nav_cols[1]:
    # 💡 [로고 2] 
    st.image("ggamlogo.png", use_container_width=True)

with nav_cols[2]:
   pass

def nav_button(label, target_page):
    is_active = (st.session_state.page == target_page)
    
    display_label = f"🔹 {label}" if is_active else label
    
    if st.button(display_label, key=f"btn_{label}", use_container_width=True, type="secondary"):
        set_page(target_page)
        st.rerun()

    if is_active:
        st.markdown("<div style='height: 3px; background-color: #2563eb; margin-top: -15px; border-radius: 2px;'></div>", unsafe_allow_html=True)

with nav_cols[3]: nav_button("홈", "홈")
with nav_cols[4]: nav_button("예측", "예측")
with nav_cols[5]: nav_button("통계", "통계")

st.divider()

# CSS 주입: 모든 버튼 또는 특정 스타일의 버튼 글자 크기 변경
st.markdown("""
    <style>
    /* 네비게이션 바 버튼 내부의 텍스트 스타일 정의 */
    div.stButton > button p {
        font-size: 20px !important; /* 원하는 크기로 조절하세요 */
        font-weight: bold !important; /* 강조하고 싶다면 추가 */
    }
    
    /* 버튼 자체의 높이도 글자 크기에 맞춰 조절하고 싶을 때 */
    div.stButton > button {
        height: 3em !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# --- 5. 페이지 렌더링 ---
# =========================================================

# ---------------------------------------------------------
# [화면 1] 홈 화면
# ---------------------------------------------------------
if st.session_state.page == '홈':
    st.markdown("""
    <div class='hero-container' style='text-align: center;'>
        <h1 style='font-size: 3.0rem; font-weight: bold; margin-bottom: 10px;'>
            당신의 수면, 뇌 건강의 신호입니다
        </h1>
        <h3 style='font-size: 1.6rem; font-weight: 500; color: #4B5563; margin-bottom: 35px;'>
            🌙 당신의 밤은 안녕하십니까?
        </h3>
        <p style='font-size: 1.2rem; color: gray; margin-bottom: 50px;'>
            공공 의료 빅데이터와 머신러닝 알고리즘을 통해<br>
            수면장애와 신경퇴행성 질환의 연관성을 분석하고 맞춤형 가이드를 확인하세요.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if st.button("🌕 예측하기", type="primary", use_container_width=True):
            set_page('예측')
            st.rerun()

# ---------------------------------------------------------
# [화면 2] AI 뇌 건강 예측 (기존 코드 완벽 보존)
# ---------------------------------------------------------
elif st.session_state.page == '예측':
    st.header("🧠 머신러닝 기반 뇌질환 위험도 예측")
    st.write("수면장애와 신경퇴행성 질환의 연관성을 분석하기 위해 공공데이터 기반의 AI 모델을 구축했습니다. 입력해주신 수면 증상을 통해 뇌 건강 위험도를 객관적으로 분석해 드립니다.")
    st.divider()

    # 진행 단계를 저장하는 session_state 초기화
    if 'pred_step' not in st.session_state:
        st.session_state.pred_step = 1

    # --- [1단계] 사용자 정보 입력 ---
    if st.session_state.pred_step == 1:
        st.subheader("1단계: 👤 사용자 정보 입력")
        
        # 이전 입력값을 기억하기 위한 초기화
        if 'gender' not in st.session_state:
            st.session_state.gender = "남성"
        if 'birth' not in st.session_state:
            st.session_state.birth = date(1974, 1, 1)

        gender = st.radio("성별", ["남성", "여성"], horizontal=True, index=0 if st.session_state.gender == "남성" else 1)
        birth = st.date_input(
            "생년월일",
            value=st.session_state.birth,
            min_value=date(1900, 1, 1),
            max_value=date.today()
        )
        age_code, real_age = get_age_code(birth)
        st.info(f"현재 나이: **만 {real_age}세**")
        
        # 다음 단계로 넘어가기
        if st.button("다음 ➡️", use_container_width=True):
            st.session_state.gender = gender
            st.session_state.birth = birth
            st.session_state.age_code = age_code
            st.session_state.pred_step = 2
            st.rerun()

    # --- [2단계] 수면 증상 체크 ---
    elif st.session_state.pred_step == 2:
        st.subheader("2단계: 💤 수면 증상 체크")
        st.write("해당하는 증상에 체크해 주세요.")
        
        c1, c2 = st.columns(2)
        with c1:
            inso  = st.checkbox("1. 잠들기 어렵거나 자다 깨면 다시 잠들기 힘들다.")
            hyper = st.checkbox("2. 밤에 충분히 잤음에도 낮에 참을 수 없이 졸음이 쏟아진다.")
            rhy   = st.checkbox("3. 교대 근무 등으로 자고 일어나는 시간이 불규칙하다.")
        with c2:
            apnea = st.checkbox("4. 자다가 숨을 멈추거나 '컥'하며 깨는 일이 잦다.")
            narco = st.checkbox("5. 웃거나 화날 때 갑자기 몸에 힘이 빠지며 잠이 든다.")
            beha  = st.checkbox("6. 꿈속 행동을 그대로 따라하며 잠꼬대나 몸부림이 심하다.")
        unspec = st.checkbox("7. 기타 설명하기 어려운 수면 불편감이 있다.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ 이전", use_container_width=True):
                st.session_state.pred_step = 1
                st.rerun()
        with col2:
            if st.button("🚀 제출 및 분석하기", type="primary", use_container_width=True):
                # 입력받은 증상 저장
                st.session_state.symptoms = {
                    'inso': inso, 'hyper': hyper, 'rhy': rhy, 
                    'apnea': apnea, 'narco': narco, 'beha': beha, 'unspec': unspec
                }
                
                matched_codes = []
                if inso:  matched_codes.append('G470')
                if hyper: matched_codes.append('G471')
                if rhy:   matched_codes.append('G472')
                if apnea: matched_codes.append('G473')
                if narco: matched_codes.append('G474')
                if beha:  matched_codes.append('G478')
                if unspec:matched_codes.append('G479')
                st.session_state.matched_codes = matched_codes
                
                # 3단계 로딩 화면으로 이동
                st.session_state.pred_step = 3
                st.rerun()

    # --- [3단계] 대기 화면 (Loading) ---
    elif st.session_state.pred_step == 3:
        st.subheader("⏳ 머신러닝 분석 중...")
        # spinner를 통해 대기 UI 렌더링
        with st.spinner("입력하신 정보를 바탕으로 뇌 건강 위험도를 분석하고 있습니다..."):
            time.sleep(2) # 2초간 대기하는 효과 (필요에 따라 시간 조절)
            st.session_state.pred_step = 4
            st.rerun()

    # --- [4단계] 결과 출력 ---
    elif st.session_state.pred_step == 4:
        st.subheader("🔍 분석 결과 리포트")
        
        # 저장해둔 session_state 값 불러오기
        gender = st.session_state.gender
        age_code = st.session_state.age_code
        symp = st.session_state.symptoms
        matched_codes = st.session_state.matched_codes
        
        if model is not None and best_threshold is not None:
            g_val = 1 if gender == "남성" else 2
            features = pd.DataFrame(
                [[g_val, age_code, int(symp['inso']), int(symp['hyper']), int(symp['rhy']),
                  int(symp['apnea']), int(symp['narco']), int(symp['beha']), int(symp['unspec'])]],
                columns=['성별코드', '연령대코드', 'insomnia', 'hypersomnia',
                         'rhythm', 'apnea', 'narcolepsy', 'behavior', 'unspecified']
            )
            
            model_prob = model.predict_proba(features)[0][1] * 100
            gender_code = 1 if gender == "남성" else 2
            sleep_score = calc_sleep_score(matched_codes, final_df, age_code, gender_code)

            if not matched_codes:
                final_prob = model_prob
            else:
                final_prob = (model_prob * 0.8) + (sleep_score * 0.2)
            
            prob = min(final_prob, 95.0)
            threshold_percent = best_threshold * 100
            grade_label, grade_css, grade_icon = get_grade(prob, threshold_percent)

            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                st.markdown(f"""
                <div style='margin-top:12px;'>
                    <div style='font-size:1rem; font-weight:bold; margin-bottom:4px;'>
                        예측 수면 위험 등급
                    </div>
                    <div class='grade-badge grade-{grade_css}'>
                        {grade_icon} {grade_label} &nbsp;|&nbsp; 수치: {prob:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if prob >= threshold_percent: 
                    st.error("🚨 **고위험군**: 정밀 검진이 필요할 수 있습니다.")
                else:
                    st.success("✅ **안전 범위**: 현재 상태를 잘 유지해 주세요.")

            with res_c2:
                gauge_color = (
                    '#e74c3c' if prob >= (threshold_percent * 1.5) else
                    '#e67e22' if prob >= threshold_percent else
                    '#f1c40f' if prob >= (threshold_percent * 0.5) else '#2ecc71'
                )
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    number={'suffix': '%', 'font': {'size': 34}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, threshold_percent * 0.5], 'color': '#eafaf1'},
                            {'range': [threshold_percent * 0.5, threshold_percent], 'color': '#fef9e7'},
                            {'range': [threshold_percent, threshold_percent * 1.5], 'color': '#fdf2e9'},
                            {'range': [threshold_percent * 1.5, 100], 'color': '#fdedec'},
                        ],
                    },
                    title={'text': "예측 수면 위험 등급", 'font': {'size': 16}}
                ))
                fig_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            if matched_codes:
                st.divider()
                st.markdown("#### 💡 맞춤형 행동 가이드")

                grade_intros = {
                    'safe':    "현재 뚜렷한 수면 위험 신호는 없습니다. 지금의 습관을 잘 유지해 주세요.",
                    'caution': "가벼운 수면 불편함이 있습니다. 생활습관 개선으로 충분히 나아질 수 있습니다.",
                    'warning': "수면 장애가 뇌 건강에 영향을 줄 수 있는 단계입니다. 적극적인 관리가 필요합니다.",
                    'danger':  "뇌질환과 연관성이 높은 수면 증상이 복합적으로 나타나고 있습니다. 전문의 상담을 강력히 권장합니다.",
                }
                st.info(grade_intros[grade_css])

                guide_tab1, guide_tab2 = st.tabs(["🚨 지금 당장 해야 할 것", "📅 장기적으로 해야 할 것"])

                with guide_tab1:
                    for code in matched_codes:
                        g = GUIDES.get(code)
                        if not g:
                            continue
                        box_class = 'guide-box-urgent' if g['urgent'] else 'guide-box'
                        icon = '🚨' if g['urgent'] else '📌'
                        items_html = "".join(f"<li style='margin-bottom:6px;'>{item}</li>" for item in g['now'])
                        st.markdown(f"""
                        <div class='{box_class}'>
                            <b>{icon} {g['name']}</b>
                            <ul style='margin-top:8px;'>{items_html}</ul>
                        </div>
                        """, unsafe_allow_html=True)

                with guide_tab2:
                    for code in matched_codes:
                        g = GUIDES.get(code)
                        if not g:
                            continue
                        items_html = "".join(f"<li style='margin-bottom:6px;'>{item}</li>" for item in g['long'])
                        st.markdown(f"""
                        <div class='guide-box'>
                            <b>📅 {g['name']}</b>
                            <ul style='margin-top:8px;'>{items_html}</ul>
                        </div>
                        """, unsafe_allow_html=True)

                st.warning("⚠️ 위 안내는 공공 데이터 기반의 통계적 경향성입니다. 증상이 지속되거나 걱정되신다면 **신경과 전문의 상담**을 받으시기 바랍니다.")
                
            # 2. 근거 문구
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef;">
                    <h5 style="margin-top: 0; color: #495057;">💡 안내사항</h5>
                    <p style="font-size: 0.9rem; line-height: 1.6; color: #6c757d; margin-bottom: 10px;">
                        본 서비스에서 제공하는 단계별 행동 가이드는 <b>미국수면의학회(AASM)</b>의 표준 가이드라인과 
                        <b>국제 수면장애 분류(ICSD-3)</b> 등 세계적으로 공인된 임상 지침을 바탕으로 작성되었습니다. 
                        또한, <b>대한수면학회</b> 및 <b>대한수면의학회</b>의 국내 진료 지침을 반영하여 한국인의 수면 특성에 최적화된 정보를 제공합니다.
                    </p>
                    <hr style="margin: 10px 0; border: 0.5px solid #dee2e6;">
                    <div style="font-size: 0.8rem; color: #adb5bd;">
                        <b>참조 기관:</b> AASM, ICSD-3, JCSM, 대한수면학회, 대한수면의학회
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                with st.expander("💡 증상별 상세 가이드 보기"):
                    st.write("체크하신 수면 증상이 없습니다. 평소 수면 습관을 점검해 보세요.")

        else:
            st.error("AI 모델 파일을 찾을 수 없습니다. 모델 파일(.pkl)의 경로와 형식을 다시 확인해 주세요.")
        
        # 다시 검사하기 버튼
        st.divider()
        if st.button("🔄 다시 검사하기", use_container_width=True):
            st.session_state.pred_step = 1
            st.rerun()
            
# ---------------------------------------------------------
# [화면 3] 빅데이터 통계 (기존 코드 완벽 보존)
# ---------------------------------------------------------
elif st.session_state.page == '통계':
    st.markdown("### 🌙 수면 장애와 신경퇴행성 질환 상관관계 (2023 공공데이터)")

    if final_df is not None:
        code_to_name = {
            'G470': '불면증',
            'G471': '과다수면',
            'G472': '수면각성일정장애',
            'G473': '수면무호흡',
            'G474': '발작수면 및 허탈발작',
            'G478': '기타 수면장애',
            'G479': '상세불명의 수면장애'
        }

        final_result = final_df['overall'].copy()
        final_result = final_result.drop(index=['G47'], errors='ignore')

        final_result.index = final_result.index.map(lambda x: code_to_name.get(x, x))
        final_result.index.name = "수면 장애 유형"
        final_result = final_result.sort_values(by='위험도(%)', ascending=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### 📋 수면 장애 유형별 통계")
            display_df = final_result.drop(columns=['환자수'], errors='ignore')
            st.dataframe(display_df, use_container_width=True)
            st.caption("💡 환자 수가 적은 유형은 위험도 수치가 크게 튈 수 있으므로, 환자 수와 함께 해석하세요.")

        with col2:
            st.markdown("##### 📊 질환 연관 위험도 (%)")
            chart_data = final_result.reset_index().sort_values(by='위험도(%)', ascending=False)

            fig = px.bar(
                chart_data,
                x="수면 장애 유형",
                y="위험도(%)",
                color="위험도(%)",
                color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                category_orders={"수면 장애 유형": chart_data["수면 장애 유형"].tolist()},
                text="위험도(%)"
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("데이터 파일을 불러올 수 없습니다.")


# --- 하단 정보 ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>Produced by Team 깜졸깜까 | Service: Alpa-Ca</div>",
    unsafe_allow_html=True
)