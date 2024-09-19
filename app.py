import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Set the page config to open the sidebar by default
st.set_page_config(layout="wide", initial_sidebar_state="expanded")


# Cache the loading of the Excel file to avoid reloading every time "Calculate" is pressed
@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name='Sheet 1')
    df['Rx Filld Dt'] = pd.to_datetime(df['Rx Filld Dt'], errors='coerce')  # Convert date column
    return df


# File downloader function
def file_download_button(file_path, label):
    with open(file_path, "rb") as f:
        file_data = f.read()
    return st.download_button(
        label=label,
        data=file_data,
        file_name=file_path.split("/")[-1],  # Extract file name from path
        mime='application/octet-stream'
    )


# Sidebar section for downloading instruction and data files
with st.sidebar:
    st.subheader("Download Files")

    # Add download button for 'instruction.pdf'
    file_download_button('instructions.pdf', 'Download Instructions')


# Cache the computation of averages
@st.cache_data
def precompute_averages(df):
    # Convert 'Rx Filld Dt' to datetime
    df['Rx Filld Dt'] = pd.to_datetime(df['Rx Filld Dt'], errors='coerce')

    # Define the start and end dates for the comparison periods
    start_prev_period = '2023-08-01'
    end_prev_period = '2024-07-31'
    start_august_2024 = '2024-08-01'
    end_august_2024 = '2024-08-31'

    # Filter data for the previous period (Aug 2023 - Jul 2024)
    prev_data = df[(df['Rx Filld Dt'] >= start_prev_period) & (df['Rx Filld Dt'] <= end_prev_period)]

    # Filter data for August 2024
    august_data = df[(df['Rx Filld Dt'] >= start_august_2024) & (df['Rx Filld Dt'] <= end_august_2024)]

    unique_dates_prev_data = prev_data['Rx Filld Dt'].nunique()
    unique_dates_august_data = august_data['Rx Filld Dt'].nunique()

    averages = {
        'claims_avg_prev': prev_data['Clm Nbr'].count() / unique_dates_prev_data,
        'cost_avg_prev': prev_data['Clnt Ingred Cost Paid Amt'].sum() / unique_dates_prev_data,
        'days_supply_avg_prev': prev_data['Days Sply Nbr'].mean(),
        'quantity_avg_prev': prev_data['Mtrc Unit Nbr'].mean(),
        'members_avg_prev': prev_data['Mbr Id'].nunique() / unique_dates_prev_data,
        'claims_avg_august': august_data['Clm Nbr'].count() / unique_dates_august_data,
        'cost_avg_august': august_data['Clnt Ingred Cost Paid Amt'].sum() / unique_dates_august_data,
        'days_supply_avg_august': august_data['Days Sply Nbr'].mean(),
        'quantity_avg_august': august_data['Mtrc Unit Nbr'].mean(),
        'members_avg_august': august_data['Mbr Id'].nunique() / unique_dates_august_data,
        'prev_categories': prev_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].unique(),
        'new_categories_august': august_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].unique(),
    }

    return averages, prev_data, august_data


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
        # Use .dropna() inline within the calculation to ignore NaN values
        result['new_therapeutic_flag'] = np.any(
            ~np.isin(
                august_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].dropna().unique(),
                prev_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].dropna().unique()
            )
        )

    return result


def plot_comparisons(averages, prev_data, august_data, selected_flags):
    cols_per_row = 4  # Set to 4 columns per row for the layout

    # Initialize containers for new drugs and new therapeutic categories
    new_drugs_list = []
    new_therapeutic_categories_list = []

    # Check if the flags for new drugs and therapeutic categories are selected
    show_new_drugs = 'New Drug (Cost > $1000)' in selected_flags
    show_new_therapeutic_categories = 'New Therapeutic Category' in selected_flags

    # Filter out 'New Drug (Cost > $1000)' and 'New Therapeutic Category' from selected_flags for plotting
    flags_to_plot = [flag for flag in selected_flags if
                     flag not in ['New Drug (Cost > $1000)', 'New Therapeutic Category']]

    # Ensure we plot a maximum of 4 flags
    flags_to_plot = flags_to_plot[:4]  # Limit to maximum of 4 flags for plotting

    # If 'New Drug (Cost > $1000)' is selected, filter and find new drugs
    if show_new_drugs:
        new_drugs = \
            august_data[~august_data['Ndc'].isin(prev_data['Ndc']) & (august_data['Clnt Ingred Cost Paid Amt'] > 1000)][
                'Ndc'].unique()
        if len(new_drugs) > 0:
            # Convert np.int64 to Python int for display purposes
            new_drugs_list.extend([int(ndc) for ndc in new_drugs])

    if show_new_therapeutic_categories:
        # Ignore NaN values in the therapeutic category comparisons
        new_therapeutic_categories = august_data[
            ~august_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].isin(
                prev_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].dropna()
            ) & august_data['Mdspn Thrptc Clsfctn Gpi 02 Nm'].notna()  # Exclude NaN values
            ]['Mdspn Thrptc Clsfctn Gpi 02 Nm'].unique()

        if len(new_therapeutic_categories) > 0:
            new_therapeutic_categories_list.extend(new_therapeutic_categories)

    # Generate the plots for selected flags (excluding new drugs and therapeutic categories)
    cols = st.columns(len(flags_to_plot))  # Create one row with as many columns as there are flags to plot (up to 4)

    # Iterate through the flags to plot
    for col_idx, flag in enumerate(flags_to_plot):
        categories = []
        prev_value = []
        august_value = []
        title = ""

        # Handle each flag by assigning values and labels
        if flag == 'Claims Increase':
            categories.append('Claims')
            prev_value.append(averages['claims_avg_prev'])
            august_value.append(averages['claims_avg_august'])
            title = 'Claims Comparison'

        if flag == 'Cost Increase':
            categories.append('Cost')
            prev_value.append(averages['cost_avg_prev'])
            august_value.append(averages['cost_avg_august'])
            title = 'Cost Comparison'

        if flag == 'Days Supply Increase':
            categories.append('Days Supply')
            prev_value.append(averages['days_supply_avg_prev'])
            august_value.append(averages['days_supply_avg_august'])
            title = 'Days Supply Comparison'

        if flag == 'Quantity Increase':
            categories.append('Quantity')
            prev_value.append(averages['quantity_avg_prev'])
            august_value.append(averages['quantity_avg_august'])
            title = 'Quantity Comparison'

        # Create the figure in the respective column
        with cols[col_idx]:
            fig, ax = plt.subplots(figsize=(4, 3))  # Consistent size for all graphs

            bar_width = 0.35
            index = np.arange(len(categories))

            bars1 = ax.bar(index, prev_value, bar_width, label='Aug 2023 - Jul 2024')
            bars2 = ax.bar(index + bar_width, august_value, bar_width, label='August 2024')

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title(title)
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(categories)

            # Add labels directly under the bars
            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, 'Aug 2023 - Jul 2024',
                        ha='center', va='bottom', fontsize=8)

            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, 'August 2024',
                        ha='center', va='bottom', fontsize=8)

            # Display the figure
            st.pyplot(fig)

    # Create a new row for displaying new drugs and new therapeutic categories in the same row
    if show_new_drugs or show_new_therapeutic_categories:
        cols_drugs_therapeutics = st.columns(2)  # Two columns: one for drugs and one for therapeutic categories

        # Display the list of new drugs in the first column
        if show_new_drugs:
            with cols_drugs_therapeutics[0]:
                if len(new_drugs_list) > 0:
                    st.subheader("List of New Drugs (NDCs) with Cost > $1000")
                    st.write(list(set(new_drugs_list)))  # Remove duplicates and display the list
                else:
                    st.write("No new drugs with cost over $1000.")

        # Display the list of new therapeutic categories in the second column
        if show_new_therapeutic_categories:
            with cols_drugs_therapeutics[1]:
                st.subheader("List of New Therapeutic Categories")
                if len(new_therapeutic_categories_list) > 0:
                    st.write(list(set(new_therapeutic_categories_list)))  # Remove duplicates and display the list
                else:
                    st.write("No new therapeutic categories.")


# Function to save the results as a CSV
def save_results_to_csv(averages, selected_flags):
    output = io.StringIO()
    data = {
        'Metrics': [],
        'Previous Period Average': [],
        'August 2024 Average': []
    }

    if 'Claims Increase' in selected_flags:
        data['Metrics'].append('Claims')
        data['Previous Period Average'].append(averages['claims_avg_prev'])
        data['August 2024 Average'].append(averages['claims_avg_august'])

    if 'Cost Increase' in selected_flags:
        data['Metrics'].append('Cost')
        data['Previous Period Average'].append(averages['cost_avg_prev'])
        data['August 2024 Average'].append(averages['cost_avg_august'])

    if 'Days Supply Increase' in selected_flags:
        data['Metrics'].append('Days Supply')
        data['Previous Period Average'].append(averages['days_supply_avg_prev'])
        data['August 2024 Average'].append(averages['days_supply_avg_august'])

    if 'Quantity Increase' in selected_flags:
        data['Metrics'].append('Quantity')
        data['Previous Period Average'].append(averages['quantity_avg_prev'])
        data['August 2024 Average'].append(averages['quantity_avg_august'])

    if 'New Drug (Cost > $1000)' in selected_flags:
        data['Metrics'].append('Drug > $1000')
        data['Previous Period Average'].append(0)  # Placeholder for previous period
        data['August 2024 Average'].append(1 if averages['cost_avg_august'] > 1000 else 0)

    results_df = pd.DataFrame(data)
    results_df.to_csv(output, index=False)
    return output.getvalue()


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
        averages, prev_data, august_data = precompute_averages(df)

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

            # Now pass the necessary arguments to plot_comparisons
            plot_comparisons(averages, prev_data, august_data, selected_flags)

        else:
            st.warning("Please select at least one flag for the analysis.")
