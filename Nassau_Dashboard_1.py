import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
import gdown

MODEL_PATH = "random_forest_model.pkl"
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=1H_gA63_IGnrqrJhHxzvPKlurNl5YoaJw",
        MODEL_PATH, quiet=False
    )
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nassau Candy – Factory Optimization",
    page_icon="🍬",
    layout="wide"
)

@st.cache_resource
def load_models():
    rf       = pickle.load(open("random_forest_model.pkl",  "rb"))
    scaler   = pickle.load(open("scaler.pkl",               "rb"))
    encoders = pickle.load(open("label_encoders.pkl",       "rb"))
    f_map    = pickle.load(open("factory_mapping.pkl",      "rb"))
    f_coords = pickle.load(open("factory_coordinates.pkl",  "rb"))
    return rf, scaler, encoders, f_map, f_coords

@st.cache_data
def load_data():
    df  = pd.read_csv("Nassau_Candy_Cleaned.csv")
    rec = pd.read_csv("recommendations.csv")
    return df, rec

try:
    rf, scaler, encoders, factory_mapping, factory_coordinates = load_models()
    df, recommendations_df = load_data()

    df['Order Date']    = pd.to_datetime(df['Order Date'])
    df['Ship Date']     = pd.to_datetime(df['Ship Date'])
    df['Lead Time']     = (df['Ship Date'] - df['Order Date']).dt.days
    df['Month']         = df['Order Date'].dt.month
    df['Year']          = df['Order Date'].dt.year
    df['Quarter']       = df['Order Date'].dt.quarter
    df['Profit Margin'] = (df['Gross Profit'] / df['Sales']) * 100
    df['Factory'] = df['Product Name'].map(factory_mapping)

    # Get exact feature names from model
    MODEL_FEATURES = rf.feature_names_in_.tolist()

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please run the Jupyter notebook first!")
    st.stop()

all_factories       = list(factory_coordinates.keys())
available_regions   = encoders["Region"].classes_.tolist()
available_shipmodes = encoders["Ship Mode"].classes_.tolist()
all_products        = list(factory_mapping.keys())
scale_cols          = ["Sales","Cost","Gross Profit","Units",
                       "Month","Quarter","Year","Profit Margin"]

def simulate(product_name, region, ship_mode):
    product_row = df[df["Product Name"] == product_name].iloc[0]
    results = []

    for factory in all_factories:
        input_data = {
            "Ship Mode"        : encoders["Ship Mode"].transform([ship_mode])[0],
            "Country/Region"   : encoders["Country/Region"].transform([product_row["Country/Region"]])[0],
            "City"             : encoders["City"].transform([product_row["City"]])[0],
            "State/Province"   : encoders["State/Province"].transform([product_row["State/Province"]])[0],
            "Postal Code"      : encoders["Postal Code"].transform([product_row["Postal Code"]])[0],
            "Division"         : encoders["Division"].transform([product_row["Division"]])[0],
            "Region"           : encoders["Region"].transform([region])[0],
            "Product ID"       : encoders["Product ID"].transform([product_row["Product ID"]])[0],
            "Product Name"     : encoders["Product Name"].transform([product_name])[0],
            "Sales"            : float(product_row["Sales"]),
            "Units"            : float(product_row["Units"]),
            "Gross Profit"     : float(product_row["Gross Profit"]),
            "Cost"             : float(product_row["Cost"]),
            "Month"            : float(product_row["Month"]),
            "Year"             : float(product_row["Year"]),
            "Quarter"          : float(product_row["Quarter"]),
            "Profit Margin"    : float(product_row["Profit Margin"]),
            "Factory Latitude" : factory_coordinates[factory][0],
            "Factory Longitude": factory_coordinates[factory][1],
            "Cluster"          : 0,
            "Cluster Name"     : 0
        }

        inp = pd.DataFrame([input_data])

        # Keep ONLY columns the model was trained on
        inp = inp[[col for col in MODEL_FEATURES if col in inp.columns]]

        # Add any missing model features as 0
        for col in MODEL_FEATURES:
            if col not in inp.columns:
                inp[col] = 0

        # Reorder to match model
        inp = inp[MODEL_FEATURES]

        # Scale numerical columns
        scale_present = [c for c in scale_cols if c in inp.columns]
        inp[scale_present] = scaler.transform(inp[scale_present])

        pred = rf.predict(inp)[0]
        results.append({
            "Factory"                    : factory,
            "Predicted Lead Time (Days)" : round(pred, 2)
        })

    return pd.DataFrame(results).sort_values(
        "Predicted Lead Time (Days)").reset_index(drop=True)

# Header
st.title("🍬 Nassau Candy Distributor")
st.subheader("Factory Reallocation & Shipping Optimization Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("🍬 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🏭 Factory Optimizer",
    "🔄 What-If Scenario",
    "📋 Recommendations",
    "⚠️ Risk & Impact"
])

# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.header("📊 Business Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",   f"{len(df):,}")
    col2.metric("Total Sales",    f"${df['Sales'].sum():,.0f}")
    col3.metric("Avg Lead Time",  f"{df['Lead Time'].mean():.0f} days",
                delta="Order to Ship")
    col4.metric("Avg Profit Margin", f"{df['Profit Margin'].mean():.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Products by Sales")
        fig, ax = plt.subplots(figsize=(7,4))
        df.groupby("Product Name")["Sales"].sum().sort_values(
            ascending=False).head(10).plot(kind="bar", ax=ax, color="steelblue")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Sales by Ship Mode")
        fig, ax = plt.subplots(figsize=(7,4))
        df.groupby("Ship Mode")["Sales"].sum().plot(kind="bar", ax=ax, color="orange")
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Monthly Sales Trend")
        fig, ax = plt.subplots(figsize=(7,4))
        df.groupby("Month")["Sales"].sum().plot(marker="o", ax=ax, color="green")
        ax.grid(True); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Quarterly Sales")
        fig, ax = plt.subplots(figsize=(7,4))
        df.groupby("Quarter")["Sales"].sum().plot(kind="bar", ax=ax, color="purple")
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Year-wise Sales")
        fig, ax = plt.subplots(figsize=(7,4))
        df.groupby("Year")["Sales"].sum().plot(kind="bar", ax=ax, color="teal")
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col6:
        st.subheader("Top Products by Profit Margin")
        fig, ax = plt.subplots(figsize=(7,4))
        df.groupby("Product Name")["Profit Margin"].mean().sort_values(
            ascending=False).head(10).plot(kind="bar", ax=ax, color="gold")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📌 KPI Summary")
    avg_imp   = recommendations_df["Improvement (Days)"].mean()
    lt_red    = (avg_imp / recommendations_df["Current Lead Time"].mean()) * 100
    prof_stab = 100 - df["Profit Margin"].std()
    rec_cov   = (len(recommendations_df) / len(all_products)) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lead Time Reduction",     f"{lt_red:.1f}%",   delta="Operational Gain")
    k2.metric("Profit Impact Stability", f"{prof_stab:.1f}", delta="Financial Safety")
    k3.metric("Scenario Confidence",     f"{rec_cov:.0f}%",  delta="Model Reliability")
    k4.metric("Recommendation Coverage", f"{len(recommendations_df)}/{len(all_products)}")

# ── PAGE 2: FACTORY OPTIMIZER ─────────────────────────────────────────────────
elif page == "🏭 Factory Optimizer":
    st.header("🏭 Factory Optimization Simulator")
    st.markdown("Select a product to see predicted lead time across all factories.")

    col1, col2, col3 = st.columns(3)
    with col1: sel_product  = st.selectbox("Select Product",   all_products)
    with col2: sel_region   = st.selectbox("Select Region",    available_regions)
    with col3: sel_shipmode = st.selectbox("Select Ship Mode", available_shipmodes)

    if st.button("🔍 Run Simulation", type="primary"):
        with st.spinner("Simulating all factories..."):
            result = simulate(sel_product, sel_region, sel_shipmode)

        st.success("Simulation Complete!")
        st.dataframe(result, use_container_width=True)

        best_factory = result.iloc[0]["Factory"]
        best_lt      = result.iloc[0]["Predicted Lead Time (Days)"]
        curr_factory = factory_mapping.get(sel_product, "Unknown")
        curr_row     = result[result["Factory"] == curr_factory]
        curr_lt      = curr_row["Predicted Lead Time (Days)"].values[0] \
                       if not curr_row.empty else best_lt

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Factory",  curr_factory, f"{curr_lt} days")
        c2.metric("Best Factory",     best_factory, f"{best_lt} days")
        c3.metric("Improvement",      f"{round(curr_lt-best_lt,2)} days",
                  delta="Gain" if curr_lt > best_lt else "Already Optimal")

        fig, ax = plt.subplots(figsize=(10,5))
        colors = ["green" if f==best_factory else "tomato" if f==curr_factory
                  else "steelblue" for f in result["Factory"]]
        ax.bar(result["Factory"], result["Predicted Lead Time (Days)"], color=colors)
        ax.set_title(f"Lead Time by Factory – {sel_product}\nGreen=Best | Red=Current")
        ax.set_xlabel("Factory"); ax.set_ylabel("Lead Time (Days)")
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 3: WHAT-IF SCENARIO ──────────────────────────────────────────────────
elif page == "🔄 What-If Scenario":
    st.header("🔄 What-If Scenario Analysis")

    col1, col2, col3 = st.columns(3)
    with col1: sel_product  = st.selectbox("Select Product",   all_products)
    with col2: sel_region   = st.selectbox("Select Region",    available_regions)
    with col3: sel_shipmode = st.selectbox("Select Ship Mode", available_shipmodes)

    st.slider("Optimization Priority: Speed vs Profit (0=Speed, 100=Profit)", 0, 100, 50)

    if st.button("⚡ Run What-If Analysis", type="primary"):
        with st.spinner("Analyzing scenarios..."):
            result = simulate(sel_product, sel_region, sel_shipmode)

        curr_factory = factory_mapping.get(sel_product, all_factories[0])
        curr_row     = result[result["Factory"] == curr_factory]
        curr_lt      = curr_row["Predicted Lead Time (Days)"].values[0] \
                       if not curr_row.empty else result.iloc[-1]["Predicted Lead Time (Days)"]
        best_lt      = result.iloc[0]["Predicted Lead Time (Days)"]
        best_factory = result.iloc[0]["Factory"]
        improvement  = round(curr_lt - best_lt, 2)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Current Assignment")
            st.info(f"**Product:** {sel_product}\n\n**Factory:** {curr_factory}\n\n**Lead Time:** {curr_lt} days")
        with col2:
            st.subheader("✅ Recommended Assignment")
            if improvement > 0:
                st.success(f"**Product:** {sel_product}\n\n**Factory:** {best_factory}\n\n**Lead Time:** {best_lt} days\n\n**Improvement:** {improvement} days ✅")
            else:
                st.warning("✅ Current factory is already optimal!")

        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10,5))
        colors = ["green" if f==best_factory else "tomato" if f==curr_factory
                  else "steelblue" for f in result["Factory"]]
        ax.bar(result["Factory"], result["Predicted Lead Time (Days)"], color=colors)
        ax.set_title("Current (Red) vs Best (Green) vs Others (Blue)")
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("📊 Full Simulation Table")
        st.dataframe(result, use_container_width=True)

# ── PAGE 4: RECOMMENDATIONS ───────────────────────────────────────────────────
elif page == "📋 Recommendations":
    st.header("📋 Ranked Factory Reassignment Recommendations")

    reassign  = recommendations_df[recommendations_df["Should Reassign"] == "Yes"]
    no_change = recommendations_df[recommendations_df["Should Reassign"] == "No"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Products",  len(recommendations_df))
    c2.metric("Should Reassign", len(reassign),  delta="Action Needed")
    c3.metric("Already Optimal", len(no_change), delta="No Change")

    st.markdown("---")
    st.subheader("✅ Products to Reassign")
    if len(reassign) > 0:
        st.dataframe(reassign[[
            "Product","Current Factory","Recommended Factory",
            "Current Lead Time","Recommended Lead Time","Improvement (Days)"
        ]].reset_index(drop=True), use_container_width=True)
    else:
        st.success("All products are already at optimal factories!")

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(12,6))
    colors = ["green" if x > 0 else "gray" for x in recommendations_df["Improvement (Days)"]]
    ax.barh(recommendations_df["Product"],
            recommendations_df["Improvement (Days)"], color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Lead Time Improvement (Days)")
    ax.set_title("Factory Reassignment Recommendations\n(Green = Improvement Possible)")
    ax.invert_yaxis()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    csv = recommendations_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "nassau_recommendations.csv", "text/csv")

# ── PAGE 5: RISK & IMPACT ─────────────────────────────────────────────────────
elif page == "⚠️ Risk & Impact":
    st.header("⚠️ Risk & Impact Panel")

    profit_sensitivity = df.groupby("Factory").agg(
        Avg_Profit_Margin  = ("Profit Margin","mean"),
        Std_Profit_Margin  = ("Profit Margin","std"),
        Total_Gross_Profit = ("Gross Profit","sum"),
        Order_Count        = ("Order ID","count")
    ).reset_index().round(2)

    st.subheader("💰 Profit Sensitivity by Factory")
    st.dataframe(profit_sensitivity, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(profit_sensitivity["Factory"],
               profit_sensitivity["Avg_Profit_Margin"], color="green")
        ax.set_title("Avg Profit Margin by Factory")
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(profit_sensitivity["Factory"],
               profit_sensitivity["Std_Profit_Margin"], color="tomato")
        ax.set_title("Profit Margin Volatility (Risk)")
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🚨 High-Risk Reassignment Warnings")
    high_risk = recommendations_df[
        (recommendations_df["Should Reassign"] == "Yes") &
        (recommendations_df["Improvement (Days)"] < 0.5)
    ]
    if len(high_risk) > 0:
        st.warning(f"⚠️ {len(high_risk)} products have minimal improvement!")
        st.dataframe(high_risk[["Product","Current Factory",
                                "Recommended Factory","Improvement (Days)"]
                               ].reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ No high-risk reassignments detected!")

    st.markdown("---")
    st.subheader("🔴 Congested Region-Product Combinations")
    region_product = df.groupby(["Region","Product Name"]).agg(
        Avg_Lead_Time=("Lead Time","mean"),
        Order_Count=("Order ID","count")
    ).reset_index()
    congested = region_product[
        region_product["Avg_Lead_Time"] > region_product["Avg_Lead_Time"].quantile(0.75)
    ].sort_values("Avg_Lead_Time", ascending=False).head(10)

    st.dataframe(congested.reset_index(drop=True), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10,5))
    labels = congested["Region"] + " | " + congested["Product Name"]
    ax.barh(labels, congested["Avg_Lead_Time"], color="tomato")
    ax.set_xlabel("Avg Lead Time (Days)")
    ax.set_title("Top 10 Congested Region-Product Combinations")
    ax.invert_yaxis()
    plt.tight_layout(); st.pyplot(fig); plt.close()

# Footer
st.markdown("---")
st.markdown(
    "<center>🍬 Nassau Candy Distributor | Factory Optimization System | Built with Streamlit & Python</center>",
    unsafe_allow_html=True
)
