import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms as T
import json
import torch.nn.functional as F
import plotly.graph_objects as go 
import os 
import requests

# --- IMPORT DATABASE ---
from nutrition_data import get_nutritional_info

# --- CONFIG ---
NUM_CLASSES = 80
MODEL_A_PATH = 'best_model_b3.pth'
MODEL_B_PATH = 'best_model_final.pth'
CLASSES_PATH = 'classes.json'
MODEL_A_URL = "https://huggingface.co/kumarajat20/nutrivision-ai-models/resolve/main/best_model_b3.pth"
MODEL_B_URL = "https://huggingface.co/kumarajat20/nutrivision-ai-models/resolve/main/best_model_final.pth

def download_model(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

download_model(MODEL_A_URL, MODEL_A_PATH)
download_model(MODEL_B_URL, MODEL_B_PATH)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI  Food Analysis",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CSS FUNCTION ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the CSS
local_css("style.css")

# --- LOAD DATA & MODELS ---
@st.cache_data
def load_class_names():
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    return {i: name for i, name in enumerate(class_names)}

try:
    idx2label = load_class_names()
except FileNotFoundError:
    st.error(f"Error: '{CLASSES_PATH}' not found.")
    st.stop()

@st.cache_resource
def load_ensemble(device='cpu'):
    model_a = timm.create_model('efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES)
    model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=device))
    model_a.to(device).eval()

    model_b = timm.create_model('efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES)
    try:
        model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))
    except FileNotFoundError:
        model_b = model_a 
        
    model_b.to(device).eval()
    return model_a, model_b

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model_a, model_b = load_ensemble(device)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- LOGIC FUNCTIONS ---
def predict(image):
    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out_a = model_a(img_t)
        out_b = model_b(img_t)
        final_out = (out_a + out_b) / 2.0
        probs = F.softmax(final_out, dim=1)
    top3_prob, top3_idx = torch.topk(probs, 3)
    results = []
    for i in range(3):
        idx = top3_idx[0][i].item()
        results.append((idx2label[idx], top3_prob[0][i].item()))
    return results

def calculate_serving_nutrition(info):
    factor = info['serving_size_g'] / 100.0
    return {
        "Calories": int(info['calories_100g'] * factor),
        "Protein": round(info['protein_100g'] * factor, 1),
        "Fat": round(info['fat_100g'] * factor, 1),
        "Carbs": round(info['carbs_100g'] * factor, 1)
    }

def plot_nutrition_chart(nutri):
    labels = ['Protein', 'Fat', 'Carbs']
    values = [nutri['Protein'], nutri['Fat'], nutri['Carbs']]
    # Custom colors to match the theme
    colors = ['#66BB6A', '#FFCA28', '#FF7043'] 
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.6, 
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text='Macros', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

# --- UI LAYOUT ---

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/706/706164.png", width=80) 
    st.markdown("## Control Panel")
    st.write("Upload an image to get started.")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
    st.markdown("---")
    st.info("💡 **Tip:** This AI works best with clear, well-lit photos of single dishes.")

# MAIN PAGE
st.title("🍛 AI  Food Analysis")
st.markdown("#### Intelligent Calorie & Nutrition Tracking System")

if uploaded_file is None:
    # Landing Page Design
    st.markdown("""
    <div class="result-card">
        <h3>👋 Welcome to your Health Dashboard</h3>
        <p>This system utilizes advanced Computer Vision to identify <b>80+ Indian Dishes</b>.</p>
        <ul>
            <li>📸 <b>Step 1:</b> Upload a photo of your food.</li>
            <li>🤖 <b>Step 2:</b> AI identifies the dish.</li>
            <li>📊 <b>Step 3:</b> Get instant nutrition info & healthy alternatives.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    col1, col2 = st.columns([1, 1.5])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    # Left Column: Image & Prediction
    with col1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        # --- FIXED LINE BELOW (use_container_width=True) ---
        st.image(image, caption='Uploaded Meal', use_container_width=True)
        
        with st.spinner('AI is analyzing...'):
            top3 = predict(image)
            predicted_dish = top3[0][0]
            confidence = top3[0][1]
        
        st.markdown(f"### 🥘 {predicted_dish.replace('_', ' ').title()}")
        st.progress(confidence)
        st.caption(f"AI Confidence: {confidence*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Alternative Guesses
        with st.expander("Other potential matches"):
            for name, score in top3[1:]:
                st.write(f"- {name.replace('_',' ').title()}: {score*100:.1f}%")

    # Right Column: Data
    with col2:
        nutri_info = get_nutritional_info(predicted_dish)
        final_nutri = calculate_serving_nutrition(nutri_info)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📊 Nutritional Profile")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Calories", f"{final_nutri['Calories']} kcal")
            st.caption(f"Serving Size: {nutri_info['serving_size_g']}g")
            st.write(f"**Protein:** {final_nutri['Protein']}g")
            st.write(f"**Carbs:** {final_nutri['Carbs']}g")
            st.write(f"**Fat:** {final_nutri['Fat']}g")
            
        with c2:
            st.plotly_chart(plot_nutrition_chart(final_nutri), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Health Check Section
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("🥗 Health Advisory")
        
        is_healthy = "Healthy" in nutri_info['healthy_alt'] or "None" in nutri_info['healthy_alt']
        
        if is_healthy:
            st.markdown(f"""
            <div class="success-box">
                <b>✅ Excellent Choice!</b><br>
                {nutri_info['alt_reason']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <b>⚠️ Healthier Alternative: {nutri_info['healthy_alt']}</b><br>
                <i style="font-size:0.9em">{nutri_info['alt_reason']}</i>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
