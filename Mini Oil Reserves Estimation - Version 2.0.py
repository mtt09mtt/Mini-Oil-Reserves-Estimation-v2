# Notes on this version (version 2.0)
# Default distribution type used due to insufficient data points (n=3) for reliable distribution prediction

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # For calculating reserves of any given probability values

# Alias the Session State
ss = st.session_state

# Page configuration
st.set_page_config(page_title="Mini Oil Reserves Estimation - Version 2.0", page_icon="📈", layout="wide")

# Add main title
st.title("Probabilistic Oil Reserves Estimate Using Monte Carlo Simulations")

# Create some tabs
tab1, tab2, tab3 = st.tabs(["✍️ Main",  "✍️ Simulated Input Parameters", "🐫 Help/About"])

# Create some place holders
place_holder1 = st.sidebar.empty()  # Place holder for user input1
place_holder2 = st.sidebar.empty()  # Place holder for user input2

# Create default distribution types
default_distribution = {
    "Net Rock Volume": "Normal",
    "Porosity": "Beta",
    "Oil saturation": "Beta",
    "Formation volume factor": "Normal",
    "Recovery factor": "Beta",
}

# Check format of the input excel file
def check_excel_format(df):
    
    # List of required column names
    required_columns = ["Reservoir", "Parameter", "Max", "Mid", "Min", "Distribution"]
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.write(f"😩 Missing columns: {', '.join(missing_columns)}. Refer to the Help page for the specific format")
        return False
    return True

# Generate input data function
def simulate_input(min_val, mean_val, max_val, num_samples, distribution):    
    
    if distribution == 'Normal':
        # Calculate standard deviation based on min, mean, and max
        std_dev = (max_val - min_val) / 6  # Assuming 6 sigma range for normal distribution
        data = np.random.normal(loc=mean_val, scale=std_dev, size=num_samples)
        data = np.clip(data, min_val, max_val)  # Clip values to stay within min and max
        
    elif distribution == 'Triangular':
        mode_val = mean_val  # Assuming mode is equal to mean for Triangular distribution
        data = np.random.triangular(left=min_val, mode=mode_val, right=max_val, size=num_samples)
        
    elif distribution == 'Uniform':
        data = np.random.uniform(low=min_val, high=max_val, size=num_samples)
        
    elif distribution == 'Lognormal':
        # Approximate parameters of the underlying normal distribution
        sigma = (np.log(max_val) - np.log(min_val)) / 6  # Assuming 6 sigma range
        mean_normal = np.log(mean_val) - 0.5 * sigma**2  # Adjust mean for lognormal
        data = np.random.lognormal(mean=mean_normal, sigma=sigma, size=num_samples)
        data = np.clip(data, min_val, max_val)  # Clip values to stay within min and max
        
    elif distribution == 'Beta':
        # Normalize mean_val, min_val, and max_val for Beta distribution
        alpha = 2  # Arbitrary choice; adjust as needed
        beta = alpha * ((max_val - mean_val) / (mean_val - min_val))  # Adjust Beta parameter
        scaled_data = np.random.beta(alpha, beta, size=num_samples)
        if np.any((scaled_data <= 0) | (scaled_data >= 1)):
            raise ValueError("Generated Beta samples must be within (0, 1). Check the parameters.")
        data = scaled_data * (max_val - min_val) + min_val

    else:
        raise ValueError("Invalid distribution type. Choose 'Normal', 'Lognormal', 'Uniform', 'Triangular', or 'Beta'.")

    return data
        
# Get out reserves value from CCDF function
def digout_reserves(x_data, y_data, probability):

    # Ensure the data is sorted correctly
    sorted_indices = np.argsort(y_data)
    x_data_sorted = np.array(x_data)[sorted_indices]
    y_data_sorted = np.array(y_data)[sorted_indices]
    
    # Create the interpolation function
    interpolation_func = interp1d(y_data_sorted, x_data_sorted, bounds_error=False, fill_value="extrapolate")
    if (probability > 0) and (probability < 1):
        out_value = interpolation_func(probability)
        return out_value
    else:
        return -999.25
    
# Plot reserves 
def plot_result(oil_reserves, res_num):
    
    # Compute the ECDF
    oil_reserves_sorted = np.sort(oil_reserves)
    y_ecdf = np.arange(1, len(oil_reserves_sorted) + 1) / len(oil_reserves_sorted)

    # Compute the CCDF - Complementary Cumulative Distribution Function
    y_ccdf = 1 - y_ecdf
    
    # Create subtitle for plot
    if res_num < ss.numOfRes :
        plot_title = "Recoverable Oil of Reservoir " + ss.reservoir_groups[res_num]
        # plot_title = "Recoverable Oil of " + xxx
    else:
        plot_title = "Aggregated Recoverable Oil Across All Reservoirs"
    
    # Create a side by side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(9, 4))
    ax1.hist(oil_reserves, bins=50, edgecolor='k', linewidth=0.2, color='blue', alpha=0.6)
    ax2.plot(oil_reserves_sorted, y_ccdf, marker='.', markersize=1, linestyle='none', label='CCDF', color='green')
    # ax2.grid(True, color='grey', alpha=0.2, linestyle='--')
    ax2.grid(True, color='grey', alpha=0.2)
    fig.suptitle(plot_title)

    ax1.set_xlabel('Recoverable Oil (MMBO)', labelpad=10, fontsize=6, color='blue', alpha=0.6)
    ax1.set_ylabel('Frequency', labelpad=10, fontsize=6, color='blue', alpha=0.6)
    
    # The labels below are followed the ROSE
    ax2.set_xlabel('Recoverable Oil (MMBO)', labelpad=10, fontsize=6, color='green')
    ax2.set_ylabel('Cumulative Probability', labelpad=10, fontsize=6, color='green')
    
    # Set font size for axis ticks
    ax1.tick_params(axis='both', labelsize=6)
    ax2.tick_params(axis='both', labelsize=6)
    
    fig.tight_layout()
    
    st.pyplot(fig, use_container_width=True)
    
    return oil_reserves_sorted, y_ccdf

# Plot simulated input
def plot_simulated_input(data1, data2, data3, data4, data5, mytitle):
    
    fig1, (myax1, myax2, myax3, myax4, myax5) = plt.subplots(1, 5, figsize = (10, 3))
    myax1.hist(data1, bins=50, color='skyblue', edgecolor='skyblue')
    myax2.hist(data2, bins=50, color='skyblue', edgecolor='skyblue')
    myax3.hist(data3, bins=50, color='skyblue', edgecolor='skyblue')
    myax4.hist(data4, bins=50, color='skyblue', edgecolor='skyblue')
    myax5.hist(data5, bins=50, color='skyblue', edgecolor='skyblue')
    fig1.suptitle("Simulated input parameters for reservoir: " + mytitle)
    myax1.xaxis.set_tick_params(labelsize=5)
    myax1.yaxis.set_tick_params(labelsize=5)
    myax1.set_xlabel("Net Rock Volume",fontsize=5, color='green')
    myax2.xaxis.set_tick_params(labelsize=5)
    myax2.yaxis.set_tick_params(labelsize=5)
    myax2.set_xlabel("Porosity",fontsize=5, color='green')
    myax3.xaxis.set_tick_params(labelsize=5)
    myax3.yaxis.set_tick_params(labelsize=5)
    myax3.set_xlabel("Oil saturation",fontsize=5, color='green')
    myax4.xaxis.set_tick_params(labelsize=5)
    myax4.yaxis.set_tick_params(labelsize=5)
    myax4.set_xlabel("Formation volume factor",fontsize=5, color='green')
    myax5.xaxis.set_tick_params(labelsize=5)
    myax5.yaxis.set_tick_params(labelsize=5)
    myax5.set_xlabel("Recovery factor",fontsize=5, color='green')
    fig1.tight_layout() 
    st.pyplot(fig1, use_container_width=True)
    
# Function for tab1
def tab1_func():
    
    list_of_distribution_type = []
    all_reserves = []
    dataframes_1 = []
    
    for res_num, reservoir_df in enumerate(ss.dataframes):
                                
        # Assign columns of dataframe to variables 
        net_rock_volume_min, net_rock_volume_mid, net_rock_volume_max = reservoir_df.iloc[0]['Min'] * 1e6, reservoir_df.iloc[0]['Mid'] * 1e6, reservoir_df.iloc[0]['Max'] * 1e6
        porosity_min, porosity_mid, porosity_max = reservoir_df.iloc[1]['Min'], reservoir_df.iloc[1]['Mid'], reservoir_df.iloc[1]['Max']
        oil_saturation_min,  oil_saturation_mid, oil_saturation_max = reservoir_df.iloc[2]['Min'], reservoir_df.iloc[2]['Mid'], reservoir_df.iloc[2]['Max']
        fvf_min, fvf_mid, fvf_max = reservoir_df.iloc[3]['Min'], reservoir_df.iloc[3]['Mid'], reservoir_df.iloc[3]['Max']
        rf_min, rf_mid, rf_max = reservoir_df.iloc[4]['Min'], reservoir_df.iloc[4]['Mid'], reservoir_df.iloc[4]['Max']
        
        # Checking for distribution type of the user input
        if ss.input_distribution == "Default distribution":
            net_rock_volume_dist = default_distribution["Net Rock Volume"]
            porosity_dist = default_distribution["Porosity"]
            oil_saturation_dist = default_distribution["Oil saturation"]
            fvf_dist = default_distribution["Formation volume factor"]
            rf_dist = default_distribution["Recovery factor"]
        else:
            net_rock_volume_dist = reservoir_df.iloc[0]['Distribution']
            porosity_dist = reservoir_df.iloc[1]['Distribution']
            oil_saturation_dist = reservoir_df.iloc[2]['Distribution']
            fvf_dist = reservoir_df.iloc[3]['Distribution']
            rf_dist = reservoir_df.iloc[4]['Distribution']
        
        # Append list_of_distribution_type to a list
        list_of_distribution_type.append({
            "Reservoir": res_num + 1,
            "Net Rock Volume": net_rock_volume_dist,
            "Porosity": porosity_dist,
            "Oil Saturation": oil_saturation_dist,
            "Fomation Volume Factor": fvf_dist,
            "Recovery Factor": rf_dist
        })
        
        # Generate 10000 data samples - Use Min, Mid, Max and the distribution
        net_rock_volume = simulate_input(net_rock_volume_min, net_rock_volume_mid, net_rock_volume_max, 10000, net_rock_volume_dist)
        porosity = simulate_input(porosity_min, porosity_mid, porosity_max, 10000, porosity_dist)
        oil_saturation = simulate_input(oil_saturation_min,  oil_saturation_mid, oil_saturation_max, 10000, oil_saturation_dist)
        fvf = simulate_input(fvf_min, fvf_mid, fvf_max, 10000, fvf_dist)
        rf = simulate_input(rf_min, rf_mid, rf_max, 10000, rf_dist)

        # Combine the generated data into a single dataframe for each simulation
        combined_data = {
            "Net Rock Volume": net_rock_volume,
            "Porosity": porosity,
            "Oil Saturation": oil_saturation,
            "FVF": fvf,
            "RF": rf
        }
        simulated_input_df = pd.DataFrame(combined_data)
        dataframes_1.append(simulated_input_df)
        
        # Calculating reserves for this reservoir
        oiip = (net_rock_volume * 6.2898 * porosity * oil_saturation) / (fvf * 1e6)

        oil_reserves = oiip * rf
        ss.oil_reserves = oil_reserves

        # Append to the aggregated reserves list
        all_reserves.append(oil_reserves)
        
        # Plot individual reservoir reserves into tab1
        plot_result(ss.oil_reserves, res_num)
              
    # Aggregate all reserves
    total_reserves = np.sum(all_reserves, axis=0)
    
    # Plot the total_reserves and write out the reserves as a table - Use any number that greater than the actual mumber of reservoir. Here I used 7777
    x_data, y_data = plot_result(total_reserves, 7777)
    p10 = digout_reserves(x_data, y_data, 0.1)
    p50 = digout_reserves(x_data, y_data, 0.5)
    p90 = digout_reserves(x_data, y_data, 0.9)
    pmean = np.mean(total_reserves)
    
    st.subheader("Aggregated Recoverable Oil Across All Reservoirs")    
    
    data = {
        'P10': p10,
        'P50': p50,
        'P90': p90,
        'Pmean': pmean}
    
    # Convert the dictionary to a DataFrame
    df_out = pd.DataFrame(list(data.items()), columns=["Probability", "Recoverable Oil (MMBOE)"])
    
    # Ensure consistent types in the 'Recoverable Oil (MMBOE)' column
    df_out["Recoverable Oil (MMBOE)"] = df_out["Recoverable Oil (MMBOE)"].astype(float)
    
    # Format the 'Recoverable Oil (MMBOE)' column to two decimal places
    df_out["Recoverable Oil (MMBOE)"] = df_out["Recoverable Oil (MMBOE)"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    # Reset the index to remove any explicit index column
    df_out.reset_index(drop=True, inplace=True)

    # Display the DataFrame as a table in Streamlit
    st.dataframe(df_out, hide_index=True)
    
    # Create a DataFrame from the list_of_distribution_type
    distribution_df = pd.DataFrame(list_of_distribution_type)
    ss.distribution_df = distribution_df
    ss.dataframes_1 = dataframes_1
            
# Function for tab2
def tab2_func():
       
    # Show distribution type
    st.subheader("Distribution types of input parameters")
    st.dataframe(ss.distribution_df)
    
    # Show histogram
    st.subheader("Histograms of input parameters")
    for res in range(len(ss.dataframes_1)):
        
        # Plot histogram of each simulated input parameter for each reservoir
        df = ss.dataframes_1[res]
        net_rock_volume = df["Net Rock Volume"]
        porosity = df["Porosity"]
        oil_saturation = df["Oil Saturation"]
        fvf = df["FVF"]
        rf = df["RF"]
        plot_simulated_input(net_rock_volume, porosity, oil_saturation, fvf, rf, ss.reservoir_groups[res])
        
# Function for tab3
def tab3_func():
    
    # Display markdown text
    st.markdown("""
        **Mini Oil Reserves Estimation - Version 2.0 - January 2025**
        
        - Accepts input for an unlimited number of reservoirs  
        - Each reservoir is treated as an independent entity with no interdependencies 
        - Supports triangular, uniform, lognormal, normal and beta probability distributions for input parameters 
        - Excludes gas-to-oil conversion and associated gas handling 
        - Runs 10,000 iterations by default for probabilistic assessments  
        - **Detailed Inputs**: 
            - An Excel file with the first sheet named "Input", has fixed column names: "Reservoir", "Parameter", "Max", "Mid", "Min" and "Distribution"
            - Distribution keywords in input file (Please remove quotes): "Normal", "Lognormal", "Triangular", "Uniform and "Beta"
            - If "Default distribution" is selected then the "Distribution" column in the input file will be ignored
            - The default:
                + Net Rock Volume: "Normal"
                + Porosity: "Beta"
                + Oil saturation: "Beta"
                + Formation volume factor: "Normal"
                + Recovery factor: "Beta"
            - Pay attention to unit of each parameter
        - **Detailed Outputs**:  
            - Probability density function (histogram) and Cumulative probability distribution curve for each reservoir
            - And aggregated results across all reservoirs
            
        - This tool is free to use, but please use it at your own risk!
        
        **👉 Feedback: mytienthang@gmail.com**
        """)  
 
# Stat function    
def main_entry():
       
    try:
        if "dataframes" not in ss:
            # Set default input distribution as "Auto-detect"
            input_distribution = place_holder1.radio(
            "👉 Select distribution type of input data",
            ["Default distribution", "User input"],
            index=0)
            ss.input_distribution = input_distribution
            
            # Load data from excel file and store it into session state
            text_out = "👉 Upload your formatted Excel file. Refer to the Help page for the specific format"
            uploaded_file = place_holder2.file_uploader(text_out, type=["xlsx"], accept_multiple_files=False)

            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file)
                
                # Check the df
                check_file = check_excel_format(df)
                if check_file:

                    # Group the DataFrame by the "Reservoir" column
                    grouped_df = df.groupby("Reservoir")
                    
                    # Get a list of unique Reservoir names
                    reservoir_groups = list(grouped_df.groups.keys())
                    ss.reservoir_groups = reservoir_groups
                    # Create a list of DataFrames, each with a unique index based on the "Reservoir" value
                    dataframes = []
                    for reservoir, group_df in grouped_df:
                        group_df = group_df.drop("Reservoir", axis=1)  # Remove the "Reservoir" column
                        group_df.index = [reservoir] * len(group_df)
                        dataframes.append(group_df)
                    
                    # Store dataframes and number of reservoir into session state (ss)     
                    ss.dataframes = dataframes
                    ss.numOfRes = len(dataframes)
                    
                    # Remove the file uploader widgets
                    place_holder1.empty()
                    place_holder2.empty()    
            
        # Execute each tab if selected  
        with tab1:
            tab1_func()     
             
        with tab2:
            tab2_func()
            
    except Exception as e:
        # st.write("👉 Incorrect input file format. Refer to the Help page for the specific format")
        pass   
    
    with tab3:
        tab3_func()
                
if __name__ == "__main__":
    main_entry()