import streamlit as st
import pandas as pd
import os
import folium
import io
import time
import speech_recognition as sr
from streamlit_folium import st_folium
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from streamlit_mic_recorder import mic_recorder
from fpdf import FPDF 

# --- 1. MULTILINGUAL UI DICTIONARY ---
MESSAGES = {
    "English": {
        "welcome": "SmartLegal AI Assistant", "ask_crime": "Describe the incident:",
        "btn_analyze": "Analyze Case", "result_head": "Legal Analysis", 
        "punish_label": "Punishment:", "fir_tab": "FIR Draft",
        "lawyer_tab": "Consult Lawyer", "police_tab": "Police Station", 
        "voice_label": "Voice Input", "station_search": "Enter Area or Pincode:", 
        "lawyer_head": "Recommended Advocates"
    },
    "Tamil": {
        "welcome": "ஸ்மார்ட்லீகல் AI உதவியாளர்", "ask_crime": "சம்பவத்தை விவரிக்கவும்:",
        "btn_analyze": "வழக்கை ஆய்வு செய்க", "result_head": "சட்ட பகுப்பாய்வு", 
        "punish_label": "தண்டனை:", "fir_tab": "முதல் தகவல் அறிக்கை",
        "lawyer_tab": "வக்கீல் ஆலோசனை", "police_tab": "காவல் நிலையம்", 
        "voice_label": "குரல் பதிவு", "station_search": "பகுதி அல்லது பின்கோடை உள்ளிடவும்:", 
        "lawyer_head": "பரிந்துரைக்கப்பட்ட வழக்கறிஞர்கள்"
    },
    "Hindi": {
        "welcome": "स्मार्टलीगल एआई सहायक", "ask_crime": "घटना का वर्णन करें:",
        "btn_analyze": "मामले का विश्लेषण करें", "result_head": "कानूनी विश्लेषण", 
        "punish_label": "सज़ा:", "fir_tab": "प्राथमिकी प्रारूप",
        "lawyer_tab": "वकील से सलाह", "police_tab": "पुलिस स्टेशन", 
        "voice_label": "आवाज़ इनपुट", "station_search": "क्षेत्र या पिनकोड दर्ज करें:", 
        "lawyer_head": "अनुशंसित वकील"
    }
}

# --- 2. PAGE CONFIG & THEME ---
st.set_page_config(page_title="SmartLegal AI v2.0", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA LOADERS ---
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    folder = "assets"
    ipc = pd.read_csv(os.path.join(folder, 'ipc_sections.csv')).fillna("")
    bns = pd.read_csv(os.path.join(folder, 'bns_sections.csv')).fillna("")
    police = pd.read_csv(os.path.join(folder, 'police_stations_chennai.csv'))
    try:
        lawyers = pd.read_csv(os.path.join(folder, 'lawyers.csv'), on_bad_lines='skip', engine='python')
        lawyers.columns = lawyers.columns.str.strip().str.lower()
    except:
        lawyers = pd.DataFrame(columns=['name', 'phone', 'city'])
    lawyers['city'] = lawyers['city'].fillna("Unknown")
    return ipc, bns, police, lawyers

model = load_ai_model()
ipc_df, bns_df, police_df, lawyers_df = load_data()

# --- 4. SESSION STATE ---
if 'input_text' not in st.session_state: st.session_state.input_text = ""
if 'step' not in st.session_state: st.session_state.step = "INPUT"
if 'history' not in st.session_state: st.session_state.history = []

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1589829545856-d10d557cf95f?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60", 
             caption="Indian Legal Intelligence System")
    
    st.title("⚖️ SmartLegal AI")
    lang_choice = st.radio("Select Language:", ["English", "Tamil", "Hindi"])
    st.session_state.lang = lang_choice
    L = MESSAGES[lang_choice]
    
    st.divider()
    # Transition Alert
    st.markdown("""
        <div style="background-color: #ffeb3b; padding: 10px; border-radius: 5px; border-left: 5px solid #fbc02d;">
            <p style="color: black; margin: 0; font-weight: bold;">⚠️ Legal Transition Alert</p>
            <p style="color: black; font-size: 0.8rem; margin: 0;">
                BNS replaced IPC on July 1, 2024. New offenses follow BNS guidelines.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.session_state.history:
        st.write("🕒 **Recent Searches**")
        for item in st.session_state.history[-3:]:
            st.caption(item)
    
    st.divider()
    st.info("📊 **Project Info**\n\nAI Engine: MiniLM-L6\nScope: IPC to BNS Transition")

# --- 6. MAIN INTERFACE ---
if st.session_state.step == "INPUT":
    st.title(f"🚀 {L['welcome']}")
    col_in, col_mic = st.columns([4, 1])
    
    with col_mic:
        st.write(f"🎤 **{L['voice_label']}**")
        audio_data = mic_recorder(start_prompt="Record", stop_prompt="Stop", key='recorder')
        if audio_data:
            st.info("🔄 Processing Audio...")
            r = sr.Recognizer()
            try:
                with sr.AudioFile(io.BytesIO(audio_data['bytes'])) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_audio = r.record(source)
                    st.session_state.input_text = r.recognize_google(audio_audio)
                    st.success("Transcribed!")
            except: st.error("Audio unclear.")

    with col_in:
        user_description = st.text_area(L['ask_crime'], value=st.session_state.input_text, height=150)

    if st.button("🔍 " + L['btn_analyze']) and user_description:
        p_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.005)
            p_bar.progress(percent + 1)
            
        with st.spinner("Analyzing Law..."):
            offenses = ipc_df['Offense'].astype(str).tolist()
            embeddings = model.encode(offenses, convert_to_tensor=True)
            query_emb = model.encode(user_description, convert_to_tensor=True)
            search_results = util.semantic_search(query_emb, embeddings, top_k=1)
            
            if search_results:
                idx = search_results[0][0]['corpus_id']
                st.session_state.match = ipc_df.iloc[idx]
                st.session_state.final_query = user_description
                st.session_state.step = "RESULT"
                st.rerun()

elif st.session_state.step == "RESULT":
    res = st.session_state.match
    lang_map = {'English': 'en', 'Tamil': 'ta', 'Hindi': 'hi'}
    target_lang = lang_map.get(st.session_state.lang, 'en')
    translator = GoogleTranslator(source='auto', target=target_lang)
    
    disp_offense = translator.translate(res['Offense'])
    disp_desc = translator.translate(res['Description'])
    disp_punish = translator.translate(res['Punishment'])
    
    if f"{disp_offense} (Sec {res['Section']})" not in st.session_state.history:
        st.session_state.history.append(f"{disp_offense} (Sec {res['Section']})")

    st.subheader(f"⚖️ {L['result_head']}: {disp_offense}")
    
    ipc_num = "".join(filter(str.isdigit, str(res['Section'])))
    bns_match = bns_df[bns_df['Section'].astype(str).str.contains(ipc_num, na=False)].head(1)
    bns_sec = f"BNS {bns_match['Section'].values[0]}" if not bns_match.empty else "BNS 303 (General)"

    m1, m2 = st.columns(2)
    with m1: st.info(f"📜 **Old Law (IPC)**\n\nSection {res['Section']}")
    with m2: st.success(f"🆕 **New Law (BNS)**\n\n{bns_sec}")

    st.markdown("### 🔍 Case Details")
    with st.container():
        col_desc, col_pun = st.columns(2)
        with col_desc:
            st.write(f"**📝 {translator.translate('Description')}:**")
            st.write(disp_desc)
        with col_pun:
            st.warning(f"**⚠️ {L['punish_label']}**\n\n{disp_punish}")

    with st.expander("🔄 What changed in the new Law (BNS)?"):
        c1, c2 = st.columns(2)
        with c1:
            st.error("**Old IPC Framework**")
            st.caption("Punitive Centered")
        with c2:
            st.success("**New BNS Framework**")
            st.caption("Justice (Nyaya) Centered")
        st.info(f"Key Update: Under BNS, Section {res['Section']} has been modernized to prioritize faster trials.")

    st.markdown("---")
    st.write("### 📊 Quick Summary Table")
    summary_data = {
        "Category": ["Offense Name", "IPC Section", "BNS Section", "Punishment"],
        "Details": [disp_offense, f"Section {res['Section']}", bns_sec, disp_punish]
    }
    st.table(pd.DataFrame(summary_data))

    st.markdown("### 🛤️ Procedural Roadmap")
    st.markdown("""
    | Step 1: Report | Step 2: Investigation | Step 3: Trial | Step 4: Verdict |
    | :--- | :--- | :--- | :--- |
    | FIR/NCR Filed | Evidence Collection | Court Hearing | Judgment |
    """)

    st.divider()
    tab1, tab2, tab3 = st.tabs([f"📍 {L['police_tab']}", f"👨‍⚖️ {L['lawyer_tab']}", f"📄 {L['fir_tab']}"])
    
    with tab1:
        st.subheader(f"📍 {L['police_tab']}")
        area = st.text_input(L['station_search'])
        if area:
            matches = police_df[police_df['name'].str.contains(area, case=False, na=False) | police_df['pincode'].astype(str).str.contains(area, na=False)]
            if not matches.empty:
                m = folium.Map(location=[matches.iloc[0]['lat'], matches.iloc[0]['lon']], zoom_start=14)
                for _, row in matches.iterrows():
                    folium.Marker(location=[row['lat'], row['lon']], popup=row['name'], icon=folium.Icon(color='blue')).add_to(m)
                st_folium(m, width=700, height=400)
                st.dataframe(matches[['name', 'address']], use_container_width=True)
            else: st.error("No station found.")

    with tab2:
        st.write(f"### {L['lawyer_head']}")
        law_matches = lawyers_df[lawyers_df['city'].str.contains("Chennai", case=False, na=False)]
        cols = [c for c in ['name', 'phone', 'category_name', 'email'] if c in lawyers_df.columns]
        st.dataframe(law_matches[cols].head(10), use_container_width=True)

    with tab3:
        st.write(f"### {L['fir_tab']}")
        fir_body = f"OFFENSE: {res['Offense']}\nSECTIONS: IPC {res['Section']} / {bns_sec}\nDETAILS: {st.session_state.final_query}"
        st.text_area("Complaint Draft", fir_body, height=200)
        if st.button("📥 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            clean_text = fir_body.encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 10, txt=clean_text)
            pdf.output("Legal_Report.pdf")
            with open("Legal_Report.pdf", "rb") as f:
                st.download_button("Download Now", f, file_name="Report.pdf")

    if st.button("🔄 Start New Search"):
        st.session_state.step = "INPUT"
        st.session_state.input_text = ""
        st.rerun()