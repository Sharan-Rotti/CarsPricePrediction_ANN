import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import base64

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Car Price Prediction & Analysis",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODEL & DATA LOADING ---
MODEL_FILE = "car_price_predictoR (3).joblib"

@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading model: {e}")
            return None
    return None

model_pipeline = load_model(MODEL_FILE)

CAR_DATA = {
    "Maruti": {"Swift": ["Petrol", "CNG"], "Swift Dzire": ["Petrol", "Diesel"], "Alto 800": ["Petrol"], "Wagon R 1.0": ["Petrol", "CNG"], "Ciaz": ["Petrol", "Diesel"], "Ertiga": ["Petrol", "Diesel"], "Vitara Brezza": ["Petrol", "Diesel"], "Baleno": ["Petrol"], "S Cross": ["Diesel"], "Celerio": ["Petrol", "CNG"], "IGNIS": ["Petrol", "CNG"]},
    "Mahindra": {"XUV500": ["Diesel"], "Scorpio": ["Diesel"], "Thar": ["Petrol", "Diesel"], "XUV300": ["Petrol", "Diesel"], "Bolero": ["Diesel"], "Marazzo": ["Diesel"], "TUV300": ["Diesel"]},
    "Volkswagen": {"Polo": ["Petrol", "Diesel"], "Vento": ["Petrol", "Diesel"], "Ameo": ["Petrol", "Diesel"], "Jetta": ["Petrol", "Diesel"], "Passat": ["Petrol", "Diesel"], "Tiguan": ["Petrol", "Diesel"]},
    "Tata": {"Nexon": ["Petrol", "Diesel", "Electric"], "Harrier": ["Diesel"], "Tiago": ["Petrol", "CNG"], "Tigor": ["Petrol", "CNG"], "Safari": ["Diesel"], "Hexa": ["Diesel"], "PUNCH": ["Petrol"]},
    "Hyundai": {"i20": ["Petrol"], "Creta": ["Petrol", "Diesel"], "Verna": ["Petrol", "Diesel"], "VENUE": ["Petrol"], "Grand i10": ["Petrol", "CNG"], "Santro": ["Petrol", "CNG"], "Xcent": ["Petrol"], "Aura": ["Petrol"]},
    "Honda": {"City": ["Petrol"], "Amaze": ["Petrol"], "Jazz": ["Petrol"], "WR-V": ["Petrol"], "BR-V": ["Petrol"], "Civic": ["Petrol"]},
    "Ford": {"EcoSport": ["Petrol", "Diesel"], "Endeavour": ["Diesel"], "Figo": ["Petrol", "Diesel"], "Aspire": ["Petrol", "Diesel"], "Freestyle": ["Petrol"]},
    "BMW": {"3 Series": ["Petrol", "Diesel"], "5 Series": ["Petrol", "Diesel"], "X1": ["Petrol", "Diesel"], "X3": ["Petrol", "Diesel"], "X5": ["Petrol", "Diesel", "Hybrid"], "7 Series": ["Petrol", "Diesel"]},
    "Renault": {"Kwid": ["Petrol"], "Duster": ["Petrol", "Diesel"], "Triber": ["Petrol"], "Kiger": ["Petrol"], "Captur": ["Petrol", "Diesel"]},
    "MG": {"Hector": ["Petrol", "Diesel"], "Hector Plus": ["Petrol", "Diesel"], "Gloster": ["Diesel"], "ZS EV": ["Electric"]},
    "Datsun": {"redi-GO": ["Petrol"], "GO": ["Petrol"], "GO+": ["Petrol"]},
    "Nissan": {"Magnite": ["Petrol"], "Kicks": ["Petrol"], "Terrano": ["Diesel"], "Sunny": ["Petrol"], "Micra": ["Petrol"]},
    "Toyota": {"Innova Crysta": ["Diesel"], "Fortuner": ["Diesel"], "Yaris": ["Petrol"], "Glanza": ["Petrol"], "Urban Cruiser": ["Petrol"], "Corolla Altis": ["Petrol", "Hybrid"]},
    "Skoda": {"Rapid": ["Petrol"], "Octavia": ["Petrol"], "Superb": ["Petrol"], "Kushaq": ["Petrol"], "Slavia": ["Petrol"]},
    "Jeep": {"Compass": ["Petrol", "Diesel"], "Wrangler": ["Petrol"], "Meridian": ["Diesel"]},
    "KIA": {"Seltos": ["Petrol"], "Sonet": ["Petrol"], "Carnival": ["Diesel"], "Carens": ["Petrol"]},
    "Audi": {"A4": ["Petrol"], "A6": ["Petrol"], "Q3": ["Petrol"], "Q5": ["Petrol"], "Q7": ["Petrol"]},
    "Landrover": {"Range Rover Evoque": ["Petrol", "Diesel"], "Discovery Sport": ["Diesel"], "Range Rover Velar": ["Petrol"]},
    "Mercedes": {"C-Class": ["Petrol"], "E-Class": ["Petrol"], "GLC": ["Petrol"], "GLE": ["Petrol"], "S-Class": ["Petrol"]},
    "Chevrolet": {"Beat": ["Petrol"], "Cruze": ["Diesel"], "Spark": ["Petrol"], "Sail": ["Petrol"], "Enjoy": ["Diesel"]},
    "Fiat": {"Punto": ["Petrol"], "Linea": ["Petrol"]},
    "Jaguar": {"XF": ["Petrol", "Diesel"], "XE": ["Petrol"], "F-PACE": ["Petrol", "Diesel"]},
    "Mitsubishi": {"Pajero Sport": ["Diesel"]},
    "CITROEN": {"C5 Aircross": ["Petrol", "Diesel"], "C3": ["Petrol"]},
    "Mini": {"Cooper": ["Petrol"]},
    "ISUZU": {"D-MAX V-Cross": ["Diesel"]},
    "Volvo": {"XC60": ["Petrol", "Hybrid"], "XC90": ["Petrol", "Hybrid"], "S90": ["Petrol"]},
    "Porsche": {"Cayenne": ["Petrol"], "Macan": ["Petrol"]},
    "Force": {"Gurkha": ["Diesel"]}
}
ALL_BRANDS = sorted(list(CAR_DATA.keys()))

# --- NEW: Image dictionary ---
# Using placeholders. Replace these with your actual image URLs.
# --- Car Brand Images Mapping ---
BRAND_IMAGES = {
    "Audi": "images/audi.jpg",
    "Bentley": "images/bentley.jpg",
    "BMW": "images/BMWW.jpg",
    "Buick": "images/buick.jpg",
    "Cadillac": "images/cadillac.jpg",
    "Chevrolet": "images/chevrolet.jpg",
    "Chrysler": "images/chrysler.jpg",
    "Dodge": "images/dodge.jpg",
    "Fiat": "images/fiat.jpg",
    "Fisker": "images/fisker.jpg",
    "Ford": "images/ford.jpg",
    "Freightliner": "images/freightliner.jpg",
    "Genesis": "images/genesis.jpg",
    "Geo": "images/geo.jpg",
    "Honda": "images/honda.jpg",
    "Hyundai": "images/hyundai.jpg",
    "Jaguar": "images/jaguar.jpg",
    "Jeep": "images/jeep.jpg",
    "Kia": "images/kia.jpg",
    "Lamborghini": "images/lamborghini.jpg",
    "Land Rover": "images/land.jpg",
    "Lexus": "images/lexus.jpg",
    "McLaren": "images/mclaren.jpg",
    "Nissan": "images/nissan.jpg",
    "Porsche": "images/porsche.jpg",
    "Toyota": "images/toyota.jpg",
    "Volkswagen": "images/volkswagen.jpg",
    "Volvo": "images/volvo.jpg"
}




# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def generate_mock_dataset(n=2000):
    """Generate a realistic-looking mock dataset for EDA and similarity comparisons."""
    rows = []
    for _ in range(n):
        brand = np.random.choice(ALL_BRANDS)
        model = np.random.choice(list(CAR_DATA[brand].keys()))
        fuel = np.random.choice(CAR_DATA[brand][model])
        age = np.random.randint(1, 15)
        km_driven = max(1000, int(np.random.normal(60000, 40000)))
        
        price = max(0.5, 15 - (age * 0.8) - (km_driven / 20000) + np.random.normal(0, 2))
        if brand in ["BMW", "Audi", "Mercedes", "Porsche", "Jaguar", "Landrover", "Volvo"]: price *= 2.5
        
        rows.append({ "brand": brand, "model": model, "fuel": fuel, "age": age,
                      "km_driven": km_driven, "price_lakhs": round(price, 2)})
    return pd.DataFrame(rows)

def safe_predict(brand, model, age, km_driven, fuel, transmission, ownership):
    """Return prediction either from pipeline or fallback logic."""
    if model_pipeline:
        try:
            X = pd.DataFrame([{"Car_Brand": brand, "Car_Model": model, "Car_Age": age, "KM Driven": km_driven, 
                               "Fuel Type": fuel, "Transmission Type": transmission, "Ownership": ownership}])
            return float(model_pipeline.predict(X)[0])
        except Exception as e:
            st.sidebar.error(f"Model prediction error: {e}") 
            # Fallthrough to mock
    
    base = 10.0 - (age * 0.7) - (km_driven / 50000)
    if fuel == "Diesel": base += 1.2
    if transmission == "Automatic": base += 1.5
    if ownership == "Second Owner": base -= 1.0
    elif "Third" in ownership or "Fourth" in ownership: base -= 2.0
    if brand in ["BMW", "Audi", "Mercedes", "Porsche", "Jaguar", "Landrover", "Volvo"]: base += 8.0
    elif brand in ["Toyota", "Skoda", "Jeep"]: base += 3.0
    return round(max(0.5, base + np.random.normal(0, 1.0)), 2)

def create_shap_plot(inputs: dict, final_price: float):
    """Generate a visual explanation of feature impact on predicted car price."""
    
    # Convert rupees → lakhs (if needed)
    final_price_lakhs = final_price / 100000
    formatted_final_price = f"{final_price_lakhs:,.2f} Lakhs"

    # Calculate mock SHAP-like contributions
    contributions = [
        -(inputs['age'] * 0.7),
        -(inputs['km'] / 50000),
        1.2 if inputs['fuel'] == 'Diesel' else -0.3,
        1.5 if inputs['transmission'] == 'Automatic' else -0.5
    ]

    features = [
        f"Age = {inputs['age']} yrs",
        f"KM Driven = {inputs['km']/1000:.1f}k km",
        f"Fuel = {inputs['fuel']}",
        f"Transmission = {inputs['transmission']}"
    ]

    df = pd.DataFrame({
        'Feature': features,
        'Contribution': contributions
    })
    df['color'] = df['Contribution'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')

    # --- Plotly Visualization ---
    fig = px.bar(
        df,
        x='Contribution',
        y='Feature',
        orientation='h',
        title=f"<b>Feature Impact on Price</b><br>Final: {formatted_final_price}",
        text='Contribution',
        template="plotly_white"
    )

    # --- Styling ---
    fig.update_traces(
        marker_color=df['color'],
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Contribution to Price (in Lakhs)",
        margin=dict(l=60, r=30, t=60, b=40)
    )

    return fig

def page_profile():
    # --- HEADER ---
    # Using st.markdown for centered text is a good approach here.
    st.markdown("""
        <div style="text-align: center;">
            <h1>👋 Hi, I'm Sharan Kumar</h1>
                 <h3>    Aspiring Data Scientist    </h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- MAIN CONTENT (Two Columns) ---
    col1, col2 = st.columns([2, 1], vertical_alignment="center")

    with col1:
        # --- BIO ---
        st.write("""
        Dedicated to applying Deep learning and Data Science techniques to extract insights, visualize trends, 
        and deploy end-to-end analytical solutions using Python and modern ML workflows.
        """)
        st.markdown("""
           <style>
               .tech-stack-container {
                   display: flex;
                   flex-wrap: wrap;
                   gap: 10px;
                   justify-content: center;
                   margin-top: 10px;
               }
               .tech-tag {
                   background-color: #1E1E2F;
                   color: white;
                   padding: 8px 15px;
                   border-radius: 25px;
                   font-size: 15px;
                   font-weight: 500;
                   border: 1px solid #444;
                   box-shadow: 0 0 8px rgba(0,0,0,0.3);
                   transition: all 0.3s ease-in-out;
               }
               .tech-tag:hover {
                   background-color: #3C3C52;
                   transform: scale(1.05);
               }
           </style>
       
           <div class="tech-stack-container">
               <span class="tech-tag">🧠 Keras</span>
               <span class="tech-tag">🐍 Python</span>
               <span class="tech-tag">🐼 Pandas</span>
               <span class="tech-tag">📈 NumPy</span>
               <span class="tech-tag">🤖 Scikit-learn</span>
               <span class="tech-tag">📊 Matplotlib & Seaborn</span>
               <span class="tech-tag">🔍 SHAP</span>
               <span class="tech-tag">🚀 Streamlit</span>
           </div>
       """, unsafe_allow_html=True)



        # --- LINKS ---
        # Icons add a nice professional touch.
        # st.markdown("""
        # 💼 **Contact:** [LinkedIn](https://www.linkedin.com/in/sharan-kumar-rotti/) • 
        # 💻 [GitHub](https://github.com/Sharan-Rotti)
        # """)
        
        # st.markdown("---")

        st.markdown("""
           <style>
               .contact-container {
                   text-align: center;
                   margin-top: 20px;
               }
               .contact-link {
                   background-color: #1E1E2F;
                   color: white;
                   padding: 10px 18px;
                   border-radius: 25px;
                   font-size: 16px;
                   font-weight: 500;
                   margin: 5px;
                   display: inline-block;
                   text-decoration: none;
                   border: 1px solid #444;
                   box-shadow: 0 0 8px rgba(0,0,0,0.3);
                   transition: all 0.3s ease-in-out;
               }
               .contact-link:hover {
                   background-color: #3C3C52;
                   transform: scale(1.08);
               }
           </style>
       
           <div class="contact-container">
               <a class="contact-link" href="https://www.linkedin.com/in/sharan-kumar-rotti/l" target="_blank">💼 LinkedIn</a>
               <a class="contact-link" href="https://github.com/Sharan-Rotti">💻 GitHub</a>
           </div>
       """, unsafe_allow_html=True)
       
        st.markdown("<hr>", unsafe_allow_html=True)


        # --- HIGHLIGHTS (Uncommented) ---
        # This section is great for showing off your ANN and Streamlit projects.
        # st.subheader("Project Highlights")
        # st.markdown("""
        # * Built end-to-end deep learning pipelines (EDA → ANN Model → Deployment).
        # * Experienced in designing and tuning neural networks using TensorFlow and Keras.
        # * Created interactive dashboards and data apps using Streamlit and Plotly.
        # """)
        st.subheader("Project Highlights")
        st.markdown("""
        * 🚗 Developed an **Artificial Neural Network (ANN)** model to predict used car prices with high accuracy.
        * 📊 Performed **comprehensive EDA and data preprocessing** using Pandas, NumPy, Matplotlib, and Seaborn.
        * 🧠 Implemented **Deep Learning architecture** using TensorFlow and Keras with optimized hyperparameters.
        * 🔍 Applied **SHAP** for model explainability to visualize feature impact on price prediction.
        * ⚙️ Utilized **Scikit-learn** for data scaling, model evaluation, and pipeline integration.
        * 🚀 Deployed a fully interactive **Streamlit web application** for real-time car price prediction.
        * 💾 Designed a clean, production-ready workflow from **data ingestion → model training → deployment**.
        """)
       

    with col2:
      def get_base64_image(image_path):
          # Verify file exists before trying to open
          if not os.path.exists(image_path):
              st.error(f"Image file not found at: {image_path}")
              return None
          try:
              with open(image_path, "rb") as img_file:
                  return base64.b64encode(img_file.read()).decode()
          except Exception as e:
              st.error(f"Error reading image: {e}")
              return None
      
      # --- Convert your local image ---
      # Ensure the 'images' folder is in the same directory as your app.py
      image_path = "images/linked_in_pic.jpg" 
      image_base64 = get_base64_image(image_path)
      
      # --- Display egg-shaped image ---
      if image_base64: # Only display if the image was successfully loaded
          st.markdown(
              f"""
              <div style="text-align: center;">
                  <img src="data:image/jpeg;base64,{image_base64}"
                       style="width:250px; height:250px; border-radius:50%; 
                              object-fit:cover; border:5px solid #ffffff; 
                              box-shadow:0 0 15px rgba(0,0,0,0.3);">
                  <h3 style="color:white;">Alok</h3>
                  <p style="color:gray;">Aspiring Data Scientist </p>
              </div>
              """,
              unsafe_allow_html=True
          )
      else:
          # Fallback placeholder if image fails to load
          st.markdown(
              """
              <div style="text-align: center; width:250px; height:250px; border-radius:50%; 
                          background-color:#eee; border:5px solid #fff; 
                          display:flex; align-items:center; justify-content:center;
                          margin:auto; box-shadow:0 0 15px rgba(0,0,0,0.3);">
                  <span style="color:#555;">Profile Image</span>
              </div>
              """, 
              unsafe_allow_html=True
          )
    st.subheader("📬 Get in Touch")
  # (then paste the HTML code above)


def page_project():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("🎯 Project Objective")
    st.write("""
    The primary objective of this project is to solve the problem of **price ambiguity** in the used car market.  
    We aim to develop a **machine learning model** that can accurately predict the value of a second-hand car
    based on key features like **age, brand, mileage, and fuel type**.
    """)
    
    st.subheader("💼 Business Context")
    st.write("""
    The second-hand car market is **large and rapidly growing**, but it often lacks transparency compared to the new car market.  
    This uncertainty creates challenges:
    - 🚘 **Buyers** risk **overpaying** for vehicles.  
    - 🏷️ **Sellers** risk **undervaluing** their assets.  
    A reliable prediction tool empowers stakeholders with an **unbiased, data-driven price estimate**, 
    helping build trust, enabling fairer negotiations, and streamlining transactions in the automotive industry.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
def page_eda():
    st.title("📈 Exploratory Data Analysis")
    st.markdown("This section provides insights from a generated sample dataset of car listings.")
    
    df = generate_mock_dataset()

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Visualizations")
    colA, colB = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    with colA:
        if numeric_cols:
            st.markdown("**Distribution Plot**")
            sel_num = st.selectbox("Select a numeric column", numeric_cols, key="hist_num")
            fig = px.histogram(df, x=sel_num, nbins=30, title=f"Distribution of {sel_num}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    with colB:
        if cat_cols:
            st.markdown("**Category Count Plot**")
            sel_cat = st.selectbox("Select a categorical column", cat_cols, key="bar_cat")
            counts = df[sel_cat].value_counts().nlargest(15)
            fig2 = px.bar(counts, x=counts.values, y=counts.index, orientation="h", title=f"Top categories in {sel_cat}", labels={"x": "Count", "y": sel_cat}, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Relationships between Features")
    colC, colD = st.columns(2)
    with colC:
        if len(numeric_cols) >= 2:
            st.markdown("**Correlation Heatmap**")
            corr = df[numeric_cols].corr()
            fig3 = px.imshow(corr, text_auto=True, title="Correlation Matrix", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
    with colD:
        if len(numeric_cols) and len(cat_cols):
            st.markdown("**Box Plot (Numeric vs. Categorical)**")
            y_col = st.selectbox("Numeric (Y-axis)", numeric_cols, key="box_y")
            x_col = st.selectbox("Category (X-axis)", cat_cols, key="box_x")
            fig4 = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_white")
            st.plotly_chart(fig4, use_container_width=True)
def page_prediction():
    st.title("🔮 Car Price Prediction")
    st.markdown("Enter the car details below to get a price estimate.")
    
    if not model_pipeline:
        st.info("ℹ️ No trained model found — using fallback prediction logic.")

    left, right = st.columns([1, 1])
    with left:
        brand = st.selectbox("Brand", ALL_BRANDS)
        model = st.selectbox("Model", sorted(CAR_DATA[brand].keys()))
        fuel = st.selectbox("Fuel Type", CAR_DATA[brand][model])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        ownership = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner", "Fourth+ Owner"])

    with right:
        age = st.number_input("Car Age (years)", 0, 30, 4)
        km_driven = st.number_input("KM Driven", 0, 500000, 45000, step=1000)

        # --- 🆕 Show brand image after KM Driven ---
        if brand in BRAND_IMAGES:
            st.image(
                BRAND_IMAGES[brand],
                caption=f"{brand} Car Image",
                width=450,
                use_container_width=False
            )
        else:
            # Fallback placeholder image if brand not found
            st.image(
                "Car Images/placeholder.png",  # You can use your uploaded image here
                caption="Car Image Placeholder",
                width=450,
                use_container_width=False
            )

    # --- Prediction Button Section ---
    if st.button("🚀 Predict Price", use_container_width=True):
        with st.spinner("Estimating price..."):
            predicted_price = safe_predict(brand, model, age, km_driven, fuel, transmission, ownership)

        # st.markdown("---")
        # st.header("Prediction Result")

        # col_l, col_r = st.columns(2)
        # with col_l:
        #     predicted_price_lakhs = predicted_price  # If safe_predict returns in Lakhs
        #     st.metric("💰 Final Price", f"₹ {predicted_price_lakhs:,.2f} Lakhs")
        #     st.info(f"**Details:** {age} years old, {km_driven:,} km, {fuel}, {transmission}")

        # with col_r:
        #     with st.expander("See Feature Impact", expanded=True):
        #         fig_imp = create_shap_plot(
        #             {'age': age, 'km': km_driven, 'fuel': fuel, 'transmission': transmission},
        #             final_price=predicted_price_lakhs * 100000
        #         )
        #         st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.header("Prediction Result")
        col_l, col_r = st.columns(2)
        with col_l:
            predicted_price_lakhs = predicted_price / 100000


            st.metric("💰 Final Price", f"₹ {predicted_price_lakhs:,.2f} Lakhs")

            st.info(f"**Details:** {age} years old, {km_driven:,} km, {fuel}, {transmission}")
        with col_r:
            with st.expander("See Feature Impact", expanded=True):
                fig_imp = create_shap_plot({'age': age, 'km': km_driven, 'fuel': fuel, 'transmission': transmission},predicted_price)
                st.plotly_chart(fig_imp, use_container_width=True)

                
# --- 5. MAIN APP LOGIC ---
st.sidebar.markdown(
    "<h2 style='text-align: center; color: white; background-color:#111827; padding:10px; border-radius:8px;'>"
    "<b>Car Price Prediction Using ANN</b></h2>", 
    unsafe_allow_html=True
)
st.sidebar.markdown("### Navigation")
page_options = {
    "Profile": page_profile,
    "Projects": page_project,
    "EDA": page_eda,
    "Prediction": page_prediction
}
selected_page_name = st.sidebar.radio("", list(page_options.keys()))
st.sidebar.markdown("---")
page_options[selected_page_name]()

st.markdown("---")
st.caption("Built by Alok • Car Price Prediction & Analysis • Use responsibly")
