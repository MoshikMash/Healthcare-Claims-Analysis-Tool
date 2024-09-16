import streamlit as st
import pandas as pd
import numpy as np


# Cache the loading of the Excel file to avoid reloading every time "Calculate" is pressed
@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name='Sheet 1')
    df['Rx Filld Dt'] = pd.to_datetime(df['Rx Filld Dt'], errors='coerce')  # Convert date column
    return df


# Cache the computation of averages
@st.cache_data
def precompute_averages(df):
    unique_dates = df['Rx Filld Dt'].nunique()

    # Precompute daily averages and totals for the previous period
    prev_data = df[df['Rx Filld Dt'].dt.month != 8]
    august_data = df[df['Rx Filld Dt'].dt.month == 8]

    averages = {
        'claims_avg_prev': prev_data['Clm Nbr'].count() / unique_dates,
        'cost_avg_prev': prev_data['Clnt Ingred Cost Paid Amt'].sum() / unique_dates,
        'days_supply_avg_prev': prev_data['Days Sply Nbr'].mean(),
        'quantity_avg_prev': prev_data['Mtrc Unit Nbr'].mean(),
        'members_avg_prev': prev_data['Mbr Id'].nunique() / unique_dates,
        'claims_avg_august': august_data['Clm Nbr'].count() / unique_dates,
        'cost_avg_august': august_data['Clnt Ingred Cost Paid Amt'].sum() / unique_dates,
        'days_supply_avg_august': august_data['Days Sply Nbr'].mean(),
        'quantity_avg_august': august_data['Mtrc Unit Nbr'].mean(),
        'members_avg_august': august_data['Mbr Id'].nunique() / unique_dates,
        'prev_categories': prev_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].unique(),
        'new_categories_august': august_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].unique()
    }

    return averages


# Cache the comparison results based on selected flags and percentage threshold
@st.cache_data
def apply_percentage_threshold(averages, x, selected_flags):
    result = {}

    # Apply percentage threshold to precomputed averages based on selected flags
    if 'Claims Increase' in selected_flags:
        result['claims_increase_flag'] = averages['claims_avg_august'] > averages['claims_avg_prev'] * (1 + x / 100)

    if 'Cost Increase' in selected_flags:
        result['cost_increase_flag'] = averages['cost_avg_august'] > averages['cost_avg_prev'] * (1 + x / 100)

    if 'Days Supply Increase' in selected_flags:
        result['days_supply_increase_flag'] = averages['days_supply_avg_august'] > averages['days_supply_avg_prev'] * (
                    1 + x / 100)

    if 'Quantity Increase' in selected_flags:
        result['quantity_increase_flag'] = averages['quantity_avg_august'] > averages['quantity_avg_prev'] * (
                    1 + x / 100)

    if 'New Drug (Cost > $1000)' in selected_flags:
        result['new_drug_flag'] = (averages['cost_avg_august'] > 1000)

    if 'New Therapeutic Category' in selected_flags:
        result['new_therapeutic_flag'] = not np.isin(averages['new_categories_august'],
                                                     averages['prev_categories']).all()

    return result


# Streamlit app
st.title("Healthcare Claims Analysis Tool")

# Move all inputs to the sidebar
with st.sidebar:
    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    # Input for the percentage threshold
    threshold = st.number_input("Enter Percentage Threshold for Analysis (e.g., 10 for 10% increase)", min_value=0,
                                max_value=100, value=10)

    # Add checkboxes for selecting which flags to include in the analysis
    st.subheader("Select Flags to Include in Analysis")
    selected_flags = []

    if st.checkbox("Claims Increase"):
        selected_flags.append("Claims Increase")

    if st.checkbox("Cost Increase"):
        selected_flags.append("Cost Increase")

    if st.checkbox("Days Supply Increase"):
        selected_flags.append("Days Supply Increase")

    if st.checkbox("Quantity Increase"):
        selected_flags.append("Quantity Increase")

    if st.checkbox("New Drug (Cost > $1000)"):
        selected_flags.append("New Drug (Cost > $1000)")

    if st.checkbox("New Therapeutic Category"):
        selected_flags.append("New Therapeutic Category")

    # Add a button to trigger the calculation in the sidebar
    calculate = st.button("Calculate")

# Process the file and calculate flags if the file is uploaded and "Calculate" is pressed
if uploaded_file is not None:
    # Load the data only once and cache it
    df = load_data(uploaded_file)

    # If the "Calculate" button is pressed
    if calculate:
        # Precompute averages only once and cache the result
        averages = precompute_averages(df)

        # Apply the percentage threshold to the precomputed averages if flags are selected
        if selected_flags:
            comparison_results = apply_percentage_threshold(averages, threshold, selected_flags)

            # Display the results
            st.subheader("Comparison Results")
            for key, value in comparison_results.items():
                result_text = f"{key.replace('_', ' ').title()}: "
                if value:
                    st.success(result_text + "Yes")
                else:
                    st.error(result_text + "No")
        else:
            st.warning("Please select at least one flag for the analysis.")
