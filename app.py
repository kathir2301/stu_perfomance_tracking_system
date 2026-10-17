import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import re

# Page configuration
st.set_page_config(
    page_title="Student Performance Tracking System",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .student-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid #BFDBFE;
        margin-bottom: 1rem;
    }
    .stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.8rem;
        text-align: center;
    }
    .sim-card {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Student Performance Tracking & Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comprehensive Machine Learning System • Individual Student Tracking, Interactive What-If Simulator & Dataset Management</div>', unsafe_allow_html=True)

# Helper functions to update / add records in CSV
def update_student_in_csv(student_id, updated_fields):
    """Update an existing student record in student_performance.csv"""
    try:
        df = pd.read_csv('student_performance.csv')
        target_id = float(student_id)
        mask = pd.to_numeric(df['StudentID'], errors='coerce') == target_id
        
        if mask.any():
            for col, val in updated_fields.items():
                if col in df.columns:
                    df.loc[mask, col] = val
            df.to_csv('student_performance.csv', index=False)
            st.cache_data.clear()
            return True
        return False
    except Exception as e:
        st.error(f"Error updating dataset: {e}")
        return False

def add_student_to_csv(new_record):
    """Append a new student record to student_performance.csv"""
    try:
        df = pd.read_csv('student_performance.csv')
        target_id = float(new_record.get('StudentID', 0))
        existing = pd.to_numeric(df['StudentID'], errors='coerce') == target_id
        if existing.any():
            st.error(f"Student ID #{int(target_id)} already exists in the dataset. Please provide a unique Student ID.")
            return False
            
        new_df = pd.DataFrame([new_record])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv('student_performance.csv', index=False)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Error adding student: {e}")
        return False

def filter_students_by_query(df_source, query_text, exact_id_match=True):
    """
    Search and filter students by StudentID or Name with precision:
    - Numeric query (e.g. '22', '#22', 'ID 22'):
      Matches exact StudentID. NEVER matches unrelated IDs containing 22 as a substring (like 1122 or 4122).
      If exact_id_match is False and no exact match exists, allows prefix matching (IDs starting with 22).
    - Text query (e.g. 'Sarah', 'John'):
      Matches Name with exact match prioritized, then starts-with, then contains.
    """
    if not query_text or not str(query_text).strip():
        return df_source, None
        
    q = str(query_text).strip()
    q_lower = q.lower()
    
    # Regex to test for ID patterns: "22", "#22", "id 22", "id: 22"
    id_match = re.match(r'^(?:id[:\s]*|#\s*)?(\d+(?:\.\d+)?)$', q, re.IGNORECASE)
    
    if id_match:
        target_id = float(id_match.group(1))
        # 1. Exact ID match
        exact_matches = df_source[pd.to_numeric(df_source['StudentID'], errors='coerce') == target_id]
        if not exact_matches.empty:
            return exact_matches, f"Exact match found for Student ID #{int(target_id)}"
            
        if exact_id_match:
            # Strictly exact match requested -> return empty with clear feedback
            return df_source.iloc[0:0], f"No student found with Student ID #{int(target_id)}"
        else:
            # Prefix search allowed: IDs starting with target_id (e.g. 22 -> 2208, 2229; never 1122)
            int_str = str(int(target_id))
            prefix_mask = df_source['StudentID'].apply(lambda x: str(int(x)).startswith(int_str) if pd.notnull(x) else False)
            prefix_matches = df_source[prefix_mask]
            
            # Also check Name in case name contains the query
            name_mask = df_source['Name'].astype(str).str.lower().str.contains(q_lower)
            name_matches = df_source[name_mask]
            
            combined = pd.concat([prefix_matches, name_matches]).drop_duplicates()
            if not combined.empty:
                return combined, f"No exact match for ID #{int(target_id)}. Showing {len(combined)} student(s) with ID starting with '{int_str}' or matching name"
            return df_source.iloc[0:0], f"No student found with Student ID #{int(target_id)}"
    else:
        # Name / text search
        name_str = df_source['Name'].astype(str)
        exact_name = df_source[name_str.str.lower() == q_lower]
        starts_name = df_source[name_str.str.lower().str.startswith(q_lower)]
        contains_name = df_source[name_str.str.lower().str.contains(q_lower)]
        
        combined = pd.concat([exact_name, starts_name, contains_name]).drop_duplicates()
        if not combined.empty:
            return combined, f"Found {len(combined)} student(s) matching name '{q}'"
        return df_source.iloc[0:0], f"No student found matching name '{q}'"

# Load data
@st.cache_data
def load_raw_and_cleaned_data():
    raw_df = pd.read_csv('student_performance.csv')
    
    # Process cleaned dataframe
    df_clean = raw_df.copy()
    
    # Clean names
    df_clean['Name'] = df_clean['Name'].fillna('Unknown Student')
    raw_df['Name'] = raw_df['Name'].fillna('Unknown Student')
    
    # Assign distinct IDs to null StudentIDs (do not fill with median!)
    if 'StudentID' in df_clean.columns:
        null_mask = df_clean['StudentID'].isnull()
        if null_mask.any():
            max_id = int(df_clean['StudentID'].dropna().max()) if df_clean['StudentID'].notnull().any() else 10000
            new_ids = [float(x) for x in range(max_id + 1, max_id + 1 + null_mask.sum())]
            df_clean.loc[null_mask, 'StudentID'] = new_ids
            raw_df.loc[null_mask, 'StudentID'] = new_ids
            
    # Fill numerical missing values (excluding StudentID)
    num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != 'StudentID']
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # Clean anomalies
    if 'Study Hours' in df_clean.columns:
        df_clean['Study Hours'] = df_clean['Study Hours'].apply(lambda x: max(0.0, float(x)))
    if 'Attendance (%)' in df_clean.columns:
        df_clean['Attendance (%)'] = df_clean['Attendance (%)'].apply(lambda x: min(100.0, float(x)))
        
    # Categorical handling
    df_clean['Gender'] = df_clean['Gender'].fillna('Male')
    df_clean['ParentalSupport'] = df_clean['ParentalSupport'].fillna('Medium')
    df_clean['Online Classes Taken'] = df_clean['Online Classes Taken'].fillna(False)
    
    # Target categorization
    def categorize(score):
        if score >= 85: return 'Good'
        elif score >= 70: return 'Average'
        else: return 'Needs Improvement'
        
    df_clean['Performance'] = df_clean['FinalGrade'].apply(categorize)
    raw_df['Performance'] = raw_df['FinalGrade'].apply(categorize)
    
    # Create ML-ready encoded dataset
    ml_df = df_clean.drop(columns=['StudentID', 'Name'], errors='ignore')
    ml_df['Gender'] = ml_df['Gender'].map({'Male': 1, 'Female': 0})
    ml_df['ParentalSupport'] = ml_df['ParentalSupport'].map({'Low': 0, 'Medium': 1, 'High': 2})
    ml_df['Online Classes Taken'] = ml_df['Online Classes Taken'].astype(int)
    
    return raw_df, df_clean, ml_df


raw_df, clean_df, ml_df = load_raw_and_cleaned_data()

# Calculate cohort class averages for benchmarking
cohort_averages = {
    'AttendanceRate': clean_df['AttendanceRate'].mean(),
    'StudyHoursPerWeek': clean_df['StudyHoursPerWeek'].mean(),
    'PreviousGrade': clean_df['PreviousGrade'].mean(),
    'FinalGrade': clean_df['FinalGrade'].mean(),
    'Study Hours': clean_df['Study Hours'].mean(),
    'Attendance (%)': clean_df['Attendance (%)'].mean(),
}

# Train Model
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['FinalGrade', 'Performance'])
    y = data['Performance']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = train_model(ml_df)

# Global feedback message banner from previous action
if 'status_toast' in st.session_state:
    st.success(st.session_state.pop('status_toast'))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Individual Student Tracking & Simulator",
    "🔮 Real-Time Student Predictor", 
    "📁 Dataset Explorer & Record Editor", 
    "📊 Academic Analytics & Insights"
])

# -------------------------------------------------------------
# TAB 1: INDIVIDUAL STUDENT TRACKING & DIAGNOSTICS & SIMULATOR
# -------------------------------------------------------------
with tab1:
    st.subheader("🎯 Individual Student Performance Tracker & Report Card")
    st.write("Track any individual student's historical records, compare their metrics against the class average, diagnose growth areas, and run interactive **What-If simulations** with direct database saving.")
    
    # Format student display strings and sort cleanly by StudentID
    clean_df['Student_Display'] = clean_df.apply(
        lambda r: f"ID #{int(r['StudentID'])} - {r['Name']}", axis=1
    )
    sorted_df = clean_df.sort_values(by='StudentID', ascending=True)
    all_student_displays = sorted_df['Student_Display'].tolist()
    
    # Search and Filter Controls
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_kw = st.text_input(
            "🔍 Search Student (ID or Name):",
            placeholder="Type ID (e.g. 22, 1, 2824) or Name (e.g. Sarah, John)...",
            key="tab1_search_kw"
        )
    with search_col2:
        exact_id_mode = st.checkbox(
            "Exact ID Match", 
            value=True, 
            help="When checked, numeric searches only match the exact Student ID (prevents partial matches like 1122 for 22).",
            key="tab1_exact_id"
        )
                
    # Filter students based on query
    matched_df, search_msg = filter_students_by_query(sorted_df, search_kw, exact_id_match=exact_id_mode)
    
    if search_kw:
        if matched_df.empty:
            st.warning(f"❌ {search_msg or f'No student found matching \"{search_kw}\"'}. Showing default student below:")
            current_options = all_student_displays
        else:
            st.info(f"ℹ️ {search_msg}")
            current_options = matched_df['Student_Display'].tolist()
    else:
        current_options = all_student_displays
        
    chosen_display = st.selectbox(
        f"Select Student to Track & Simulate ({len(current_options)} available):",
        current_options,
        index=0,
        key=f"tab1_student_select_{len(current_options)}"
    )
    
    # Extract selected student row
    matched_rows = clean_df[clean_df['Student_Display'] == chosen_display]
    student_record = matched_rows.iloc[0] if not matched_rows.empty else clean_df.iloc[0]
    
    # 1. Header Student Card
    s_name = str(student_record['Name'])
    s_id = int(student_record['StudentID']) if pd.notnull(student_record['StudentID']) else 1
    s_gender = str(student_record['Gender'])
    s_parental = str(student_record['ParentalSupport'])
    s_online = "Yes" if student_record['Online Classes Taken'] else "No"
    s_tier = str(student_record['Performance'])
    s_final = float(student_record['FinalGrade'])
    
    tier_colors = {
        'Good': ('#2ecc71', '#D4EDDA', '#155724'),
        'Average': ('#f39c12', '#FFF3CD', '#856404'),
        'Needs Improvement': ('#e74c3c', '#F8D7DA', '#721C24')
    }
    t_dot, t_bg, t_fg = tier_colors.get(s_tier, ('#3B82F6', '#DBEAFE', '#1E3A8A'))
    
    st.markdown(f"""
    <div style="background:{t_bg}; color:{t_fg}; padding:1.2rem; border-radius:0.75rem; border:1px solid {t_dot}; margin-bottom:1.2rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h2 style="margin:0; font-size:1.6rem; color:{t_fg};">🧑‍🎓 {s_name} (Student ID: #{s_id})</h2>
                <p style="margin:4px 0 0 0; font-size:0.95rem; opacity:0.9;">
                    <b>Gender:</b> {s_gender} &nbsp;|&nbsp; 
                    <b>Parental Support:</b> {s_parental} &nbsp;|&nbsp; 
                    <b>Online Classes:</b> {s_online} &nbsp;|&nbsp; 
                    <b>Extracurricular Activities:</b> {int(student_record['ExtracurricularActivities'])}
                </p>
            </div>
            <div style="text-align:right;">
                <span style="background:{t_dot}; color:white; font-size:1rem; font-weight:bold; padding:0.4rem 0.9rem; border-radius:20px;">
                    {s_tier.upper()}
                </span>
                <p style="margin:4px 0 0 0; font-size:0.85rem;">Final Exam Grade: <b>{s_final}</b></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Key Metrics with Delta vs Class Average
    m1, m2, m3, m4 = st.columns(4)
    
    att_diff = student_record['AttendanceRate'] - cohort_averages['AttendanceRate']
    study_diff = student_record['StudyHoursPerWeek'] - cohort_averages['StudyHoursPerWeek']
    prev_diff = student_record['PreviousGrade'] - cohort_averages['PreviousGrade']
    final_diff = student_record['FinalGrade'] - cohort_averages['FinalGrade']
    
    m1.metric("Attendance Rate", f"{student_record['AttendanceRate']:.1f}%", f"{att_diff:+.1f}% vs Class Avg")
    m2.metric("Weekly Study Hours", f"{student_record['StudyHoursPerWeek']:.1f} hrs", f"{study_diff:+.1f} hrs vs Class Avg")
    m3.metric("Previous Term Grade", f"{student_record['PreviousGrade']:.1f}", f"{prev_diff:+.1f} vs Class Avg")
    m4.metric("Final Grade Score", f"{student_record['FinalGrade']:.1f}", f"{final_diff:+.1f} vs Class Avg")
    
    st.markdown("---")
    
    # 3. Side-by-Side Visual Comparison & AI Diagnostic Analysis
    chart_col, diag_col = st.columns([1.2, 1])
    
    with chart_col:
        st.write(f"##### 📊 {s_name} vs Class Benchmark Comparison")
        
        comp_df = pd.DataFrame({
            'Metric': ['Attendance Rate (%)', 'Weekly Study Hours', 'Previous Term Grade', 'Daily Study Hours', 'Final Grade'],
            f'{s_name}': [
                student_record['AttendanceRate'],
                student_record['StudyHoursPerWeek'],
                student_record['PreviousGrade'],
                student_record['Study Hours'],
                student_record['FinalGrade']
            ],
            'Class Average': [
                cohort_averages['AttendanceRate'],
                cohort_averages['StudyHoursPerWeek'],
                cohort_averages['PreviousGrade'],
                cohort_averages['Study Hours'],
                cohort_averages['FinalGrade']
            ]
        })
        
        comp_melted = comp_df.melt(id_vars='Metric', var_name='Type', value_name='Value')
        fig_comp, ax_comp = plt.subplots(figsize=(7, 4.2))
        sns.barplot(data=comp_melted, x='Metric', y='Value', hue='Type', palette=['#3B82F6', '#9CA3AF'], ax=ax_comp)
        plt.xticks(rotation=20, ha='right', fontsize=9)
        plt.ylabel("Score / Hours / %")
        plt.title(f"Individual Tracking: {s_name} vs Cohort Benchmark", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_comp)
        
    with diag_col:
        st.write(f"##### 🔍 Academic Diagnostic & Action Plan")
        
        strengths = []
        concerns = []
        
        if student_record['AttendanceRate'] >= cohort_averages['AttendanceRate']:
            strengths.append(f"Strong class attendance ({student_record['AttendanceRate']:.0f}% is above class average of {cohort_averages['AttendanceRate']:.0f}%).")
        else:
            concerns.append(f"Attendance rate ({student_record['AttendanceRate']:.0f}%) is below the {cohort_averages['AttendanceRate']:.0f}% class benchmark.")
            
        if student_record['StudyHoursPerWeek'] >= cohort_averages['StudyHoursPerWeek']:
            strengths.append(f"Dedicated weekly study discipline ({student_record['StudyHoursPerWeek']:.0f} hrs/wk).")
        else:
            concerns.append(f"Weekly study time ({student_record['StudyHoursPerWeek']:.0f} hrs/wk) is below class average ({cohort_averages['StudyHoursPerWeek']:.0f} hrs/wk).")
            
        if student_record['ParentalSupport'] == 'High':
            strengths.append("High home and parental support environment.")
        elif student_record['ParentalSupport'] == 'Low':
            concerns.append("Low parental support reported; benefits from school-based mentoring.")
            
        if student_record['PreviousGrade'] >= 80:
            strengths.append("Solid academic background from previous semesters.")
            
        st.write("**💪 Key Strengths:**")
        if strengths:
            for s in strengths:
                st.write(f"• {s}")
        else:
            st.write("• Consistent participation in regular academic coursework.")
            
        st.write("**⚠️ Areas for Improvement:**")
        if concerns:
            for c in concerns:
                st.write(f"• {c}")
        else:
            st.write("• No critical risk factors detected. Maintain current study discipline!")
            
        st.write("**💡 Tailored Mentor Recommendation:**")
        if s_tier == 'Good':
            st.info(f"Keep up the great work! {s_name} is performing at an honors level. Recommended for peer leadership or advanced problem sets.")
        elif s_tier == 'Average':
            st.warning(f"Encourage {s_name} to add 3–4 hours of weekly study and maintain attendance ≥ 90% to reach the **Good** tier.")
        else:
            st.error(f"High Priority: Schedule academic counseling for {s_name}. Implement a daily study schedule and monitor weekly attendance.")

    # 4. Interactive & Updatable "What-If" Improvement Simulator
    st.markdown("---")
    st.write(f"### 🔮 Interactive What-If Simulator & Goal Optimizer for {s_name}")
    st.caption("Test hypothetical improvements in study habits, attendance, and support. Apply automated AI goals, or commit the simulated changes directly to the permanent student dataset.")
    
    # Manage session state for simulator widgets per student
    student_state_key = f"active_tracker_id"
    if st.session_state.get(student_state_key) != s_id:
        st.session_state[student_state_key] = s_id
        st.session_state[f"sim_study_{s_id}"] = float(student_record['StudyHoursPerWeek'])
        st.session_state[f"sim_att_{s_id}"] = float(student_record['AttendanceRate'])
        st.session_state[f"sim_extra_{s_id}"] = min(int(student_record['ExtracurricularActivities']), 3)
        st.session_state[f"sim_parental_{s_id}"] = s_parental
        st.session_state[f"sim_online_{s_id}"] = bool(student_record['Online Classes Taken'])
        st.session_state[f"sim_study_day_{s_id}"] = float(student_record['Study Hours'])
        st.session_state[f"sim_prev_{s_id}"] = float(student_record['PreviousGrade'])

    # AI Goal Seeker Algorithm
    def find_optimal_goal(target_tier="Good"):
        best_cfg = None
        min_extra_study = 999
        
        sim_gender_num = 1 if s_gender == 'Male' else 0
        p_sup_num = {'Low': 0, 'Medium': 1, 'High': 2}.get(s_parental, 1)
        onl_num = 1 if student_record['Online Classes Taken'] else 0
        
        # Grid search over realistic attendance and study steps
        for cand_att in [float(student_record['AttendanceRate']), 85.0, 90.0, 95.0, 98.0]:
            for cand_study in [float(student_record['StudyHoursPerWeek']), 15.0, 20.0, 25.0, 30.0, 35.0]:
                test_row = pd.DataFrame([{
                    'Gender': sim_gender_num,
                    'AttendanceRate': max(float(student_record['AttendanceRate']), cand_att),
                    'StudyHoursPerWeek': max(float(student_record['StudyHoursPerWeek']), cand_study),
                    'PreviousGrade': float(student_record['PreviousGrade']),
                    'ExtracurricularActivities': float(student_record['ExtracurricularActivities']),
                    'ParentalSupport': p_sup_num,
                    'Study Hours': max(float(student_record['Study Hours']), cand_study / 5.0),
                    'Attendance (%)': max(float(student_record['Attendance (%)']), cand_att),
                    'Online Classes Taken': onl_num
                }])
                pred = model.predict(test_row)[0]
                prob = np.max(model.predict_proba(test_row)[0])
                if pred == target_tier:
                    added_effort = (cand_study - student_record['StudyHoursPerWeek']) + 0.5 * (cand_att - student_record['AttendanceRate'])
                    if added_effort < min_extra_study:
                        min_extra_study = added_effort
                        best_cfg = (cand_study, cand_att, prob)
        return best_cfg

    # Goal Seeker Box
    with st.expander("🎯 AI Target Goal Optimizer (Auto-Calculate Path to Top Tier)", expanded=False):
        opt_col1, opt_col2 = st.columns([2, 1])
        target_choice = "Good"
        optimal_res = find_optimal_goal("Good")
        
        with opt_col1:
            if optimal_res:
                rec_study, rec_att, rec_prob = optimal_res
                delta_st = rec_study - student_record['StudyHoursPerWeek']
                delta_at = rec_att - student_record['AttendanceRate']
                st.markdown(f"""
                **Optimal Roadmap for {s_name} to reach `GOOD` Performance:**
                - 📚 **Recommended Weekly Study:** `{rec_study:.1f} hrs/week` ({f'+{delta_st:.1f} hrs' if delta_st > 0 else 'Current level'})
                - 🏫 **Recommended Attendance:** `{rec_att:.1f}%` ({f'+{delta_at:.1f}%' if delta_at > 0 else 'Current level'})
                - 🎯 **Projected Model Confidence:** `{rec_prob*100:.1f}%`
                """)
            else:
                st.info(f"{s_name} is already performing at or near optimal target conditions.")
                
        with opt_col2:
            if optimal_res and st.button("🚀 Apply AI Goal to Simulator", use_container_width=True):
                st.session_state[f"sim_study_{s_id}"] = float(optimal_res[0])
                st.session_state[f"sim_att_{s_id}"] = float(optimal_res[1])
                st.rerun()

    # Simulator Input Controls
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_study = st.slider(
            f"Simulate Weekly Study Hours:", 
            min_value=1.0, max_value=40.0, 
            value=float(st.session_state.get(f"sim_study_{s_id}", student_record['StudyHoursPerWeek'])), 
            step=0.5,
            key=f"slider_study_{s_id}"
        )
    with sim_c2:
        sim_att = st.slider(
            f"Simulate Attendance Rate (%):", 
            min_value=50.0, max_value=100.0, 
            value=float(st.session_state.get(f"sim_att_{s_id}", student_record['AttendanceRate'])), 
            step=1.0,
            key=f"slider_att_{s_id}"
        )
    with sim_c3:
        cur_extra = min(int(st.session_state.get(f"sim_extra_{s_id}", student_record['ExtracurricularActivities'])), 3)
        sim_extra = st.selectbox(
            f"Simulate Extracurriculars:", 
            [0, 1, 2, 3], 
            index=cur_extra,
            key=f"select_extra_{s_id}"
        )
        
    # Advanced Fine-Tuning Factors
    with st.expander("⚙️ Additional Simulation Factors (Parental Support, Daily Hours, Online Classes, Previous Grade)", expanded=False):
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        with adv_col1:
            p_opts = ["Low", "Medium", "High"]
            cur_p = st.session_state.get(f"sim_parental_{s_id}", s_parental)
            p_idx = p_opts.index(cur_p) if cur_p in p_opts else 1
            sim_parental = st.selectbox("Parental Support:", p_opts, index=p_idx, key=f"select_p_{s_id}")
        with adv_col2:
            sim_study_day = st.slider(
                "Daily Study Hours:", 0.0, 8.0, 
                float(st.session_state.get(f"sim_study_day_{s_id}", student_record['Study Hours'])), 
                0.5, key=f"slider_day_{s_id}"
            )
        with adv_col3:
            sim_prev = st.slider(
                "Previous Term Grade:", 50.0, 100.0, 
                float(st.session_state.get(f"sim_prev_{s_id}", student_record['PreviousGrade'])), 
                1.0, key=f"slider_prev_{s_id}"
            )
        with adv_col4:
            sim_online = st.checkbox(
                "Attended Online Classes", 
                value=bool(st.session_state.get(f"sim_online_{s_id}", student_record['Online Classes Taken'])),
                key=f"check_online_{s_id}"
            )

    # Encode simulated input
    sim_gender_num = 1 if s_gender == 'Male' else 0
    sim_support_num = {'Low': 0, 'Medium': 1, 'High': 2}.get(sim_parental, 1)
    sim_online_num = 1 if sim_online else 0
    
    sim_input = pd.DataFrame([{
        'Gender': sim_gender_num,
        'AttendanceRate': float(sim_att),
        'StudyHoursPerWeek': float(sim_study),
        'PreviousGrade': float(sim_prev),
        'ExtracurricularActivities': float(sim_extra),
        'ParentalSupport': sim_support_num,
        'Study Hours': float(sim_study_day),
        'Attendance (%)': float(sim_att),
        'Online Classes Taken': sim_online_num
    }])
    
    # Baseline input for side-by-side comparison
    base_input = pd.DataFrame([{
        'Gender': sim_gender_num,
        'AttendanceRate': float(student_record['AttendanceRate']),
        'StudyHoursPerWeek': float(student_record['StudyHoursPerWeek']),
        'PreviousGrade': float(student_record['PreviousGrade']),
        'ExtracurricularActivities': float(student_record['ExtracurricularActivities']),
        'ParentalSupport': {'Low': 0, 'Medium': 1, 'High': 2}.get(s_parental, 1),
        'Study Hours': float(student_record['Study Hours']),
        'Attendance (%)': float(student_record['Attendance (%)']),
        'Online Classes Taken': 1 if student_record['Online Classes Taken'] else 0
    }])
    
    # Model predictions
    base_pred = model.predict(base_input)[0]
    base_probs = model.predict_proba(base_input)[0]
    
    sim_pred = model.predict(sim_input)[0]
    sim_probs = model.predict_proba(sim_input)[0]
    classes = model.classes_
    
    # Visual Outcome Display
    tier_badges = {
        'Good': ('#2ecc71', '🟢 GOOD PERFORMANCE'),
        'Average': ('#f39c12', '🟡 AVERAGE PERFORMANCE'),
        'Needs Improvement': ('#e74c3c', '🔴 NEEDS IMPROVEMENT')
    }
    
    st.write("")
    out_c1, out_c2 = st.columns([1.3, 1])
    
    with out_c1:
        st.markdown(f"""
        <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:0.75rem; padding:1.2rem;">
            <h4 style="margin:0 0 8px 0; color:#166534;">📈 Projected Simulation Outcome for {s_name}</h4>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                <span style="font-size:1.1rem; font-weight:600; color:#374151;">Baseline: <code>{base_pred.upper()}</code></span>
                <span style="font-size:1.3rem;">➔</span>
                <span style="background:{tier_badges.get(sim_pred, ('#3B82F6', ''))[0]}; color:white; font-size:1.1rem; font-weight:bold; padding:4px 12px; border-radius:16px;">
                    {sim_pred.upper()}
                </span>
            </div>
            <p style="margin:0; color:#15803D; font-size:0.95rem;">
                <b>Model Confidence:</b> <b>{np.max(sim_probs)*100:.1f}%</b> &nbsp;|&nbsp; 
                <b>Study Hours Delta:</b> {sim_study - student_record['StudyHoursPerWeek']:+.1f} hrs &nbsp;|&nbsp; 
                <b>Attendance Delta:</b> {sim_att - student_record['AttendanceRate']:+.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with out_c2:
        # Comparison Probability Chart
        prob_df = pd.DataFrame({
            'Tier': classes,
            'Baseline (%)': [round(p * 100, 1) for p in base_probs],
            'Simulated (%)': [round(p * 100, 1) for p in sim_probs]
        }).melt(id_vars='Tier', var_name='Scenario', value_name='Probability (%)')
        
        fig_prob, ax_prob = plt.subplots(figsize=(5.5, 2.7))
        sns.barplot(data=prob_df, x='Tier', y='Probability (%)', hue='Scenario', palette=['#9CA3AF', '#3B82F6'], ax=ax_prob)
        plt.ylim(0, 100)
        plt.title("Probability Shift (Baseline vs. Simulated)", fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_prob)

    # Action Buttons: Save & Reset
    st.write("")
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1.5, 1.5])
    
    with btn_col1:
        if st.button("💾 Save & Commit Changes to Student Record (CSV)", type="primary", use_container_width=True):
            # Calculate estimated final grade adjustment if performance changed
            new_final = float(student_record['FinalGrade'])
            if sim_pred == 'Good' and new_final < 85.0:
                new_final = 86.0
            elif sim_pred == 'Average' and (new_final < 70.0 or new_final >= 85.0):
                new_final = 75.0
            elif sim_pred == 'Needs Improvement' and new_final >= 70.0:
                new_final = 65.0
                
            updated_data = {
                'StudyHoursPerWeek': float(sim_study),
                'AttendanceRate': float(sim_att),
                'ExtracurricularActivities': float(sim_extra),
                'ParentalSupport': sim_parental,
                'Study Hours': float(sim_study_day),
                'Attendance (%)': float(sim_att),
                'PreviousGrade': float(sim_prev),
                'Online Classes Taken': bool(sim_online),
                'FinalGrade': float(new_final)
            }
            
            success = update_student_in_csv(s_id, updated_data)
            if success:
                st.session_state['status_toast'] = f"✅ Successfully updated and saved record for {s_name} (ID: #{s_id}) in student_performance.csv!"
                st.rerun()
                
    with btn_col2:
        if st.button("🔄 Reset Simulator to Original Values", use_container_width=True):
            st.session_state[f"sim_study_{s_id}"] = float(student_record['StudyHoursPerWeek'])
            st.session_state[f"sim_att_{s_id}"] = float(student_record['AttendanceRate'])
            st.session_state[f"sim_extra_{s_id}"] = min(int(student_record['ExtracurricularActivities']), 3)
            st.session_state[f"sim_parental_{s_id}"] = s_parental
            st.session_state[f"sim_study_day_{s_id}"] = float(student_record['Study Hours'])
            st.session_state[f"sim_prev_{s_id}"] = float(student_record['PreviousGrade'])
            st.session_state[f"sim_online_{s_id}"] = bool(student_record['Online Classes Taken'])
            st.rerun()

# -------------------------------------------------------------
# TAB 2: REAL-TIME PREDICTOR FOR NEW STUDENTS
# -------------------------------------------------------------
with tab2:
    st.subheader("👤 Real-Time Student Profile Predictor")
    st.write("Input custom parameters for any hypothetical or new incoming student to generate instantaneous performance predictions and actionable mentor guidance.")
    
    student_name = st.text_input("Enter Student Full Name:", value="Alex Morgan", help="Student's name will appear on the generated performance card.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        attendance_rate = st.slider("Attendance Rate (%)", min_value=50.0, max_value=100.0, value=88.0, step=1.0, key="p_att_rate")
        study_hours_week = st.slider("Weekly Study Hours", min_value=1.0, max_value=40.0, value=20.0, step=1.0, key="p_study_wk")
        
    with col2:
        previous_grade = st.slider("Previous Term Grade", min_value=50.0, max_value=100.0, value=80.0, step=1.0, key="p_prev_grd")
        extracurriculars = st.selectbox("Extracurricular Activities Count", [0, 1, 2, 3], index=1, key="p_extra")
        parental_support = st.selectbox("Parental Support Level", ["Low", "Medium", "High"], index=2, key="p_supp")
        
    with col3:
        study_hours_daily = st.slider("Daily Study Hours", min_value=0.0, max_value=8.0, value=3.0, step=0.5, key="p_study_day")
        attendance_pct = st.slider("Overall Attendance (%)", min_value=50.0, max_value=100.0, value=85.0, step=1.0, key="p_att_pct")
        online_classes = st.checkbox("Attended Online Classes", value=True, key="p_online")
        
    st.write("")
    if st.button("🚀 Predict Performance for " + (student_name if student_name else "Student"), use_container_width=True, type="primary"):
        gender_num = 1 if gender == "Male" else 0
        support_map = {"Low": 0, "Medium": 1, "High": 2}
        support_num = support_map[parental_support]
        online_num = 1 if online_classes else 0
        
        input_data = pd.DataFrame([{
            'Gender': gender_num,
            'AttendanceRate': float(attendance_rate),
            'StudyHoursPerWeek': float(study_hours_week),
            'PreviousGrade': float(previous_grade),
            'ExtracurricularActivities': float(extracurriculars),
            'ParentalSupport': support_num,
            'Study Hours': float(study_hours_daily),
            'Attendance (%)': float(attendance_pct),
            'Online Classes Taken': online_num
        }])
        
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        st.markdown("---")
        
        st.markdown(f"""
        <div class="student-card">
            <h3 style="margin:0; color:#1E3A8A;">📋 Student Performance Report: <u>{student_name}</u></h3>
            <p style="margin-top:4px; color:#4B5563; font-size:0.95rem;">
                <b>Gender:</b> {gender} &nbsp;|&nbsp; 
                <b>Attendance Rate:</b> {attendance_rate}% &nbsp;|&nbsp; 
                <b>Study Hours/Week:</b> {study_hours_week} hrs &nbsp;|&nbsp; 
                <b>Previous Grade:</b> {previous_grade} &nbsp;|&nbsp; 
                <b>Parental Support:</b> {parental_support}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1.2, 1.8])
        
        with res_col1:
            if prediction == "Good":
                st.success(f"### 🎉 Result: {prediction} Performance")
                st.info("🟢 **Academic Tier:** On track for Honors / Excellent standing.")
            elif prediction == "Average":
                st.warning(f"### ⚡ Result: {prediction} Performance")
                st.info("🟡 **Academic Tier:** Meeting standard curriculum requirements.")
            else:
                st.error(f"### ⚠️ Result: {prediction}")
                st.info("🔴 **Academic Tier:** At-risk student. Early intervention recommended.")
                
        with res_col2:
            st.write("**Model Prediction Confidence Breakdown:**")
            conf_df = pd.DataFrame({
                'Performance Category': classes, 
                'Confidence (%)': [round(p * 100, 1) for p in probs]
            })
            st.bar_chart(conf_df.set_index('Performance Category'), color='#3B82F6')
            
        st.markdown("#### 💡 Mentor Action Plan & Recommendations:")
        if prediction == "Good":
            st.write(f"✅ **For {student_name}:** Excellent study discipline! Encourage continuation of current study routines and peer mentoring opportunities.")
        elif prediction == "Average":
            st.write(f"💡 **For {student_name}:** Solid baseline performance. Increasing weekly study hours by 3–5 hours and maintaining ≥ 90% attendance will elevate results to **Good**.")
        else:
            st.write(f"🚨 **For {student_name}:** Recommended for immediate academic intervention: 1-on-1 tutoring sessions, weekly progress reviews, and structured study schedules.")

# -------------------------------------------------------------
# TAB 3: DATASET EXPLORER & RECORD EDITOR
# -------------------------------------------------------------
with tab3:
    st.subheader("📁 Complete Student Records & Dataset Management")
    st.write("Explore, search, filter, and modify student performance records in the live dataset.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", f"{len(raw_df):,}")
    m2.metric("Good Performance", f"{(clean_df['Performance'] == 'Good').sum()}")
    m3.metric("Average Performance", f"{(clean_df['Performance'] == 'Average').sum()}")
    m4.metric("Needs Improvement", f"{(clean_df['Performance'] == 'Needs Improvement').sum()}")
    
    # Direct Editor Section
    with st.expander("✏️ Quick Record Editor & Student Creator", expanded=False):
        edit_mode = st.radio("Action:", ["Edit Existing Student Record", "Add New Student Record"], horizontal=True)
        
        if edit_mode == "Edit Existing Student Record":
            ed_source = clean_df.sort_values(by='StudentID', ascending=True)
            ed_c1, ed_c2 = st.columns([2.5, 1])
            with ed_c1:
                ed_query = st.text_input("🔍 Find Student to Edit (ID or Name):", placeholder="e.g. 22, Sarah, 2824...", key="ed_search_input")
            with ed_c2:
                ed_exact = st.checkbox("Exact ID Match", value=True, key="ed_exact_checkbox")
                
            if ed_query:
                ed_matched, ed_msg = filter_students_by_query(ed_source, ed_query, exact_id_match=ed_exact)
                if ed_matched.empty:
                    st.warning(f"❌ {ed_msg or f'No student found matching \"{ed_query}\"'}")
                    ed_options = ed_source['Student_Display'].tolist()
                else:
                    st.info(f"ℹ️ {ed_msg}")
                    ed_options = ed_matched['Student_Display'].tolist()
            else:
                ed_options = ed_source['Student_Display'].tolist()
                
            ed_chosen = st.selectbox("Select Student to Edit:", ed_options, key=f"ed_selector_{len(ed_options)}")
            ed_rec = clean_df[clean_df['Student_Display'] == ed_chosen].iloc[0]
            ed_id = int(ed_rec['StudentID']) if pd.notnull(ed_rec['StudentID']) else 1
            
            with st.form("edit_student_form"):
                e_c1, e_c2, e_c3 = st.columns(3)
                with e_c1:
                    e_name = st.text_input("Name:", value=str(ed_rec['Name']))
                    e_gender = st.selectbox("Gender:", ["Male", "Female"], index=0 if ed_rec['Gender'] == 'Male' else 1)
                    e_att_rate = st.number_input("Attendance Rate (%):", min_value=0.0, max_value=100.0, value=float(ed_rec['AttendanceRate']), step=0.5)
                with e_c2:
                    e_study_wk = st.number_input("Weekly Study Hours:", min_value=0.0, max_value=60.0, value=float(ed_rec['StudyHoursPerWeek']), step=0.5)
                    e_prev = st.number_input("Previous Grade:", min_value=0.0, max_value=100.0, value=float(ed_rec['PreviousGrade']), step=0.5)
                    e_final = st.number_input("Final Grade:", min_value=0.0, max_value=100.0, value=float(ed_rec['FinalGrade']), step=0.5)
                with e_c3:
                    e_extra = st.selectbox("Extracurriculars:", [0, 1, 2, 3], index=min(int(ed_rec['ExtracurricularActivities']), 3))
                    p_idx = ["Low", "Medium", "High"].index(ed_rec['ParentalSupport']) if ed_rec['ParentalSupport'] in ["Low", "Medium", "High"] else 1
                    e_parental = st.selectbox("Parental Support:", ["Low", "Medium", "High"], index=p_idx)
                    e_online = st.checkbox("Online Classes Taken", value=bool(ed_rec['Online Classes Taken']))
                    
                submitted = st.form_submit_button("💾 Update Student Record in Dataset", type="primary", use_container_width=True)
                if submitted:
                    up_dict = {
                        'Name': e_name,
                        'Gender': e_gender,
                        'AttendanceRate': float(e_att_rate),
                        'StudyHoursPerWeek': float(e_study_wk),
                        'PreviousGrade': float(e_prev),
                        'FinalGrade': float(e_final),
                        'ExtracurricularActivities': float(e_extra),
                        'ParentalSupport': e_parental,
                        'Online Classes Taken': e_online,
                        'Study Hours': float(e_study_wk / 5.0),
                        'Attendance (%)': float(e_att_rate)
                    }
                    if update_student_in_csv(ed_id, up_dict):
                        st.session_state['status_toast'] = f"✅ Updated record for {e_name} (ID: #{ed_id}) successfully!"
                        st.rerun()
        else:
            with st.form("add_student_form"):
                a_c1, a_c2, a_c3 = st.columns(3)
                next_id = int(clean_df['StudentID'].max() + 1) if 'StudentID' in clean_df.columns else len(clean_df) + 1
                with a_c1:
                    a_id = st.number_input("Student ID (Unique Number):", min_value=1, max_value=999999, value=next_id, step=1)
                    a_name = st.text_input("Full Name:", value="New Student")
                    a_gender = st.selectbox("Gender:", ["Male", "Female"])
                    a_att = st.number_input("Attendance Rate (%):", min_value=0.0, max_value=100.0, value=85.0, step=0.5)
                with a_c2:
                    a_study = st.number_input("Weekly Study Hours:", min_value=0.0, max_value=60.0, value=15.0, step=0.5)
                    a_prev = st.number_input("Previous Grade:", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
                    a_final = st.number_input("Final Grade:", min_value=0.0, max_value=100.0, value=78.0, step=0.5)
                with a_c3:
                    a_extra = st.selectbox("Extracurriculars:", [0, 1, 2, 3], index=1)
                    a_parental = st.selectbox("Parental Support:", ["Low", "Medium", "High"], index=1)
                    a_online = st.checkbox("Online Classes Taken", value=True)
                    
                add_submitted = st.form_submit_button("➕ Add Student to Database", type="primary", use_container_width=True)
                if add_submitted:
                    new_rec = {
                        'StudentID': float(a_id),
                        'Name': a_name,
                        'Gender': a_gender,
                        'AttendanceRate': float(a_att),
                        'StudyHoursPerWeek': float(a_study),
                        'PreviousGrade': float(a_prev),
                        'ExtracurricularActivities': float(a_extra),
                        'ParentalSupport': a_parental,
                        'FinalGrade': float(a_final),
                        'Study Hours': float(a_study / 5.0),
                        'Attendance (%)': float(a_att),
                        'Online Classes Taken': a_online
                    }
                    if add_student_to_csv(new_rec):
                        st.session_state['status_toast'] = f"✅ Added student {a_name} (ID: #{int(a_id)}) successfully!"
                        st.rerun()

    st.write("")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 0.9, 1.1, 1.1])
    
    with filter_col1:
        search_query = st.text_input("🔍 Search by Student Name or ID:", placeholder="e.g. 22, John, Sarah, 2824...", key="ds_search")
        
    with filter_col2:
        ds_exact_id = st.checkbox("Exact ID", value=True, help="When checked, numeric ID searches match only the exact Student ID.", key="ds_exact_id")
        
    with filter_col3:
        tier_filter = st.selectbox("Filter by Performance:", ["All Categories", "Good", "Average", "Needs Improvement"], key="ds_tier")
        
    with filter_col4:
        support_filter = st.selectbox("Filter by Parental Support:", ["All", "High", "Medium", "Low"], key="ds_supp")
        
    display_df = clean_df.copy().drop(columns=['Student_Display'], errors='ignore')
    
    if search_query:
        display_df, ds_msg = filter_students_by_query(display_df, search_query, exact_id_match=ds_exact_id)
        if ds_msg:
            if display_df.empty:
                st.warning(f"❌ {ds_msg}")
            else:
                st.info(f"ℹ️ {ds_msg}")
        
    if tier_filter != "All Categories":
        display_df = display_df[display_df['Performance'] == tier_filter]
        
    if support_filter != "All":
        display_df = display_df[display_df['ParentalSupport'] == support_filter]
        
    st.write(f"Showing **{len(display_df)}** matching student records:")
    st.dataframe(display_df, use_container_width=True, height=400)
    
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Student Dataset (CSV)",
        data=csv_data,
        file_name="filtered_student_performance.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# TAB 4: DATA ANALYTICS & INSIGHTS
# -------------------------------------------------------------
with tab4:
    st.subheader("📊 Academic Analytics & Factors Influencing Performance")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("##### 1. Distribution of Performance Tiers")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        order = ['Good', 'Average', 'Needs Improvement']
        palette = {'Good': '#2ecc71', 'Average': '#f39c12', 'Needs Improvement': '#e74c3c'}
        sns.countplot(data=clean_df, x='Performance', order=order, palette=palette, ax=ax1, hue='Performance', legend=False)
        ax1.set_ylabel("Number of Students")
        ax1.set_xlabel("Performance Category")
        for p in ax1.patches:
            h = int(p.get_height())
            ax1.annotate(f"{h}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
        st.pyplot(fig1)
        
    with col_b:
        st.write("##### 2. Feature Importance Ranking")
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        importances.plot(kind='barh', color='#3B82F6', ax=ax2)
        ax2.set_xlabel("Relative Importance Score")
        ax2.set_ylabel("Features")
        for p in ax2.patches:
            w = p.get_width()
            ax2.annotate(f"{w*100:.1f}%", (w, p.get_y() + p.get_height()/2),
                         ha='left', va='center', xytext=(4, 0), textcoords='offset points', fontsize=9)
        st.pyplot(fig2)
