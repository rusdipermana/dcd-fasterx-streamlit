#from hashlib import file_digest
import pandas as pd
import streamlit as st
import plotly.express as px
import time
import matplotlib.pyplot as plt
from torch import NoneType
import app_component as ac
import calendar
import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go

# Matikan peringatan savefig
st.set_option('deprecation.showPyplotGlobalUse', False)

def read_sequence(uploaded_file):
    # Read Excel file into a DataFrame
    data = pd.read_excel(uploaded_file)
    data.columns = data.iloc[0]
    data_index = data.iloc[1:].reset_index(drop=True)
    subset_kolom = ['VIN', 'Model Series', 'Model Name', 'MY', 'Country', 'Order No.', 'Dom./Exp.',
                    'W-on Date', 'W-on Shift', 'W-on SEQ', 'W-off Line', 'W-off Date', 'W-off Shift', 'W-off SEQ']

    # Select specific columns from the DataFrame
    data_cleaned = data_index[subset_kolom]

    # Convert columns in subset_kolom to string
    for column in subset_kolom:
        data_cleaned.loc[:, column] = data_cleaned[column].astype(str).str.strip()

    # Convert data types
    data_cleaned.loc[:, 'W-on Date'] = pd.to_datetime(data_cleaned['W-on Date'], format='%Y%m%d')
    data_cleaned.loc[:, 'W-off Date'] = pd.to_datetime(data_cleaned['W-off Date'], format='%Y%m%d')
    print("check tipe data :",data_cleaned.dtypes)


    return data_cleaned

@st.cache_resource
def filter_data_yearly(data):
    subset_kolom = ['VIN', 'Model Series', 'Model Name', 'MY', 'Country', 'Order No.', 'Dom./Exp.',
                    'W-on Date', 'W-on Shift', 'W-on SEQ', 'W-off Line', 'W-off Date', 'W-off Shift', 'W-off SEQ']

    # Select specific columns from the DataFrame
    data = data[subset_kolom]

    # Convert columns in subset_kolom to string
    #for column in subset_kolom:
        #data.loc[:, column] = data[column].astype(str).str.strip()

    # Check if the 'W-on Date' column is already in datetime format
    if 'W-on Date' in data.columns and not pd.api.types.is_datetime64_ns_dtype(data['W-on Date']):
        data['W-on Date'] = pd.to_datetime(data['W-on Date'], format='%Y%m%d')

    # Extract year and month from the date
    data['Year'] = data['W-on Date'].dt.year
    data['Month'] = data['W-on Date'].dt.month

    # Sorting data by Year and Month
    filtered_data = data.sort_values(by=['Year', 'Month'])

    # Calculate counts for each Model Series per month
    yearly_counts = filtered_data.groupby(['Model Series', 'Year', 'Month']).size().reset_index(name='Counts')

    # Combine Month and Year into a new 'Month Year' column
    yearly_counts['Month Year'] = yearly_counts.apply(lambda row: f"{calendar.month_name[row['Month']]} {str(row['Year'])[2:]}", axis=1)

    return yearly_counts



@st.cache_resource
def map_model_series_to_line(data):
    # Buat pemetaan model series
    model_series_mapping = {
        "Line 1": ["JA0", "J40", "J60"],
        "Line 2": ["J20"],
        "Line 3": ["200"]
    }

    # Buat kolom baru "Line" berdasarkan pemetaan model series
    data['Line'] = data['Model Series'].map({model: line for line, models in model_series_mapping.items() for model in models})

    # Hitung jumlah Model Series dalam setiap Line
    line_counts = data.groupby('Line')['Model Series'].nunique().reset_index()
    line_counts.columns = ['Line', 'Model Series Count']

    # Perhitungan data month-year
    data['Year'] = data['W-on Date'].dt.year
    data['Month'] = data['W-on Date'].dt.month
    data['Month Year'] = data.apply(lambda row: f"{calendar.month_name[row['Month']]} {str(row['Year'])[2:]}", axis=1)

    return line_counts, data




# Buat pemetaan model series
model_series_mapping = {
    "Line 1": ["JA0", "J40", "J60"],
    "Line 2": ["J20"],
    "Line 3": ["200"]
}

@st.cache_resource
def filter_data_date(data, start_date, end_date, selected_line=None, selected_model_series=None):
    # Convert start_date and end_date to datetime
    #start_date = pd.to_datetime(start_date, format='%Y%m%d')
    #end_date = pd.to_datetime(end_date, format='%Y%m%d')
    #print(start_date)


    subset_kolom = ['VIN', 'Model Series', 'Model Name', 'MY', 'Country', 'Order No.', 'Dom./Exp.',
                    'W-on Date', 'W-on Shift', 'W-on SEQ', 'W-off Line', 'W-off Date', 'W-off Shift', 'W-off SEQ']

    # Select specific columns from the DataFrame
    data = data[subset_kolom]

    # Convert columns in subset_kolom to string
    #for column in subset_kolom:
        #data.loc[:, column] = data[column].astype(str).str.strip()

    #print(data)
    if 'W-on Date' in data.columns and not pd.api.types.is_datetime64_ns_dtype(data['W-on Date']):
        data['W-on Date'] = pd.to_datetime(data['W-on Date'], format='%Y%m%d')
    
    print("tipe data :",data.dtypes)

    # Filter data based on the selected date range
    print("check kondisi Compare Date",((data['W-on Date'] >= start_date) & (data['W-on Date'] <= end_date)))

    filtered_data = data[(data['W-on Date'] >= start_date) & (data['W-on Date'] <= end_date)]

    #print(filtered_data)

    # Filter data berdasarkan Line yang dipilih
    if selected_line:
        valid_models = model_series_mapping.get(selected_line, [])
        filtered_data = filtered_data[filtered_data['Model Series'].isin(valid_models)]

    # Filter data berdasarkan Model Series yang dipilih
    if selected_model_series:
        filtered_data = filtered_data[filtered_data['Model Series'].isin(selected_model_series)]

    return filtered_data

# Fungsi untuk menghitung jumlah produksi berdasarkan kolom tertentu
def calculate_column_count(data, column_name):
    # Pastikan kolom yang ingin dihitung berada dalam dataframe
    if column_name in data.columns:
        # Menghitung jumlah produksi
        count = data[column_name].count()
        return count
    else:
        return None  # Kolom tidak ditemukan dalam dataframe

# Fungsi untuk menghitung persentase produksi berdasarkan kolom tertentu
def calculate_production_percentage(data, column_name):
    total_production = calculate_column_count(data, column_name)

    if total_production is not None:
        data_with_percentage = data.groupby(column_name).size().reset_index(name='Counts')
        data_with_percentage['Percentage'] = (data_with_percentage['Counts'] / total_production) * 100
        return data_with_percentage
    else:
        return None

# Initialize mapping_done as False outside the function
mapping_done = False
month_names = {i: calendar.month_name[i] for i in range(1, 13)}
data = None


def uploaded():
    global mapping_done  # Make sure we use the global variable
    global data
    #data = None

    # Generate unique keys for the radio buttons
    from_tab1, from_tab2 = st.tabs(["Upload Form", "Production Sequence System"])

    
    

    with from_tab1:
        
        data = None # Initialize data as None

        uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

        if uploaded_file is not None:
            # Display the "Baca Data" button
            if st.button("Read Data"):
                # Display a progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()  # To display status information
                test_reading = "Reading Dataset..."
                status_text.text(test_reading)

                # Read the Excel file while updating the progress bar
                data = read_sequence(uploaded_file)  # Update the 'data' variable
                for i in range(101):
                    progress_bar.progress(i)
                    time.sleep(0.1)  # Replace with the actual reading time

                test_reading = "File uploaded successfully"
                status_text.text(test_reading)
                if data is not None:  # Check if data is not None before using it

                    st.write("Total rows in uploaded data:", len(data))
                    # Mengambil baris terakhir dari data yang tidak kosong dan memiliki kolom 'VIN' yang bukan "nan"
                    
                    

                    header1, header2 = st.columns([2, 1])
                    with header1:
                        last_valid_vin_row = data[data['VIN'] != "nan"].iloc[-1]

                        if not last_valid_vin_row.empty:
                            last_valid_w_on_date = last_valid_vin_row['W-on Date']
                            formatted_w_on_date = last_valid_w_on_date.strftime("%B %d, %Y")

                            info_text = f"Last valid VIN: {last_valid_vin_row['VIN']} on {formatted_w_on_date}"

                            st.info(info_text)

                        else:
                            st.write("Tidak ada VIN yang valid dalam data.")
                    with header2:
                        with st.expander("Last Updated Date", expanded=False):
                            if not data.empty:
                                first_row = data.iloc[0]  # Assuming the first row contains the date
                                last_row = data.iloc[-1]  # Get the last row
                                formatted_date_first = first_row["W-on Date"].strftime("%B %d, %Y")
                                formatted_date_last = last_row["W-on Date"].strftime("%B %d, %Y")
                                st.write("First row - W-on Date:", f"<span style='color:red'><b>{formatted_date_first}</b></span>", unsafe_allow_html=True)
                                st.write("Last row - W-on Date:", f"<span style='color:red'><b>{formatted_date_last}</b></span>", unsafe_allow_html=True)

                    
                    
                    # Filter data for a specific year (e.g., 2023) and calculate monthly frequency
                    year_to_filter = 2023  # Change this to the desired year
                    yearly_counts = filter_data_yearly(data)

                    # Create a select box to choose "Model Series"
                    selected_model_series = st.selectbox("Select Model Series", yearly_counts['Model Series'].unique(), key="selectbox_1")

                    # Periksa apakah pemetaan bulan sudah dilakukan sebelumnya
                    if not mapping_done:
                        yearly_counts['Month'] = yearly_counts['Month'].map(month_names)
                        mapping_done = True  # Atur ke True setelah pemetaan dilakukan

                    # Filter data based on selected "Model Series"
                    selected_data = yearly_counts[yearly_counts['Model Series'] == selected_model_series]

                    # Konversi kamus ke daftar
                    category_array = list(month_names.values())
                    
                    col1, col2 = st.columns([2, 1])

                    # Create a Plotly bar chart
                    fig = px.bar(selected_data, x='Month Year', y='Counts', labels={'Counts': 'Total Production'}, title=f"Monthly Counts for {selected_model_series}")
                    # Dalam bagian plotly, gunakan category_array dalam properti categoryarray
                    fig.update_xaxes(type='category', categoryorder='array', categoryarray=category_array)
                    # Set plot to autosize to the available width
                    fig.update_layout(autosize=True)
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Display the DataFrame in the second column
                        selected_data_filtered = selected_data[['Month Year', 'Counts']]
                        selected_data_filtered = selected_data_filtered.reset_index(drop=True) 
                        st.dataframe(selected_data_filtered, use_container_width=True, height=400)

                    # Set default end date to today
                    # Initialize today_date with the current date
                    today_date = datetime.date.today()
                    #today_date =  datetime.datetime.combine(today_date, datetime.time())
                    print("todaydate :",today_date)

                    # Calculate the start date as 1 month ago
                    #one_month_ago = today_date - datetime.timedelta(days=30)

                    # Date range selection using date_input with default values
                    #start_date = st.date_input("Start Date", today_date)
                    #end_date = st.date_input("End Date", today_date)
                    
                    col3, col4 = st.columns([2, 1])

                    with col3:
                        st.write("Select date range:")


                        start_date1 = st.date_input("Start Date", today_date, key="start_date_input1")
                        #start_date = start_date + datetime.timedelta(days=2)
                        end_date1 = st.date_input("End Date", today_date, key="end_date_input1")

                        # Convert start_date to datetime.datetime
                        start_date_datetime = datetime.datetime(start_date1.year, start_date1.month, start_date1.day)
                        end_date_datetime = datetime.datetime(end_date1.year, end_date1.month, end_date1.day)
                        # Convert the selected dates to datetime.datetime objects
                        #start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
                        #end_date = datetime.datetime.combine(end_date, datetime.datetime.max.time())
                        # Format start_date
                        start_date_str = start_date_datetime.strftime('%Y%m%d')
                        #print("ini strtime :",start_date_str)
                        end_date_str = end_date_datetime.strftime('%Y%m%d')

                        #target_date = datetime.datetime(2023, 10, 20)
                        # Filter data berdasarkan tanggal
                        #target_date_str = target_date.strftime('%Y%m%d')
                        #print("ini strtime example :",target_date_str)
                        #filter_data = data.loc[data['W-on Date'] == target_date_str]
                        #print("tipe data filter_data :", filter_data.dtypes)
                        #print("filter data date :",filter_data)

                    with col4:
                        st.write("Select Model Series:")
                        selected_line = st.selectbox("Select Line", list(model_series_mapping.keys()), key="selectbox_2")
                        

                        if not start_date1 or not end_date1:
                            st.warning("Please select the date range.")
                        else:
                            # Use st.session_state to store the selected line
                            if "selected_line" not in st.session_state:
                                st.session_state.selected_line = selected_line
                            else:
                                st.session_state.selected_line = selected_line

                            # Define default values based on the selected line
                            default_model_series = model_series_mapping.get(st.session_state.selected_line, [])
                            valid_default_model_series = [value for value in default_model_series if value in model_series_mapping.get(selected_line, [])]
                            selected_model_series = st.multiselect("Select Model Series", model_series_mapping.get(selected_line, []), default=valid_default_model_series, key="multiselect1")
                            filtered_data = filter_data_date(data, start_date_str, end_date_str, selected_line, selected_model_series)
                            
                            columns_to_display = ['Model Series', 'Model Name', 'MY', 'Country', 'Dom./Exp.', 'W-on Date', 'W-on Shift', 'W-on SEQ']
                            filtered_data_display = filtered_data[columns_to_display]
                            #print(filtered_data_display)

                    col5, col6 = st.columns([2, 1])

                    with col6:
                        colOpt, colModel = st.columns([1, 1])
                        with colOpt:
                            shift_option1 = st.radio("Select Shift:", ("All Shifts", "1", "2"), key = "radio1")
                            filtered_data_display = filtered_data_display.reset_index(drop=True)    
                            # Filter data berdasarkan selected_shift
                            if shift_option1 != "All Shifts":
                                filtered_data_display = filtered_data_display[filtered_data_display['W-on Shift'] == shift_option]

                        with colModel:
                            model_series_counts = filtered_data_display["Model Series"].value_counts()
                            # Create a DataFrame to hold the counts
                            counts_df = pd.DataFrame({"Counts": model_series_counts})
                            # Add a row at the bottom with the total counts
                            counts_df.loc["Total"] = counts_df.sum()

                            model_name_counts = filtered_data_display["Model Name"].value_counts()
                            # Create a DataFrame for "Model Name" counts
                            model_name_counts_df = pd.DataFrame(model_name_counts).reset_index()
                            # Add a date column (for example, the current date)
                            model_name_counts_df["Date"] = [datetime.date.today()] * len(model_name_counts_df)

                            # Rename the columns
                            model_name_counts_df.columns = ["Model Name", "Count", "Date"]
                            st.dataframe(counts_df)
                        with st.expander("Details Type", expanded=False):
                            st.dataframe(model_name_counts_df)
                    
                    with col5:

                        
                        # Hitung jumlah masing-masing nilai dalam kolom "Country"
                        country_counts = filtered_data_display["Country"].value_counts()

                        # Buat data untuk Pie Chart
                        labels = country_counts.index.tolist()
                        values = country_counts.values.tolist()

                        # Buat Pie Chart dengan innerRadius
                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])

                        # Atur layout untuk tampilan lebih baik
                        fig.update_layout(title_text="Production by country")

                        # Tampilkan Pie Chart dalam Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                progress_bar.empty()  # Remove the progress bar
            
             # Menambahkan tombol "Save" untuk menyimpan data ke Parquet
            if data is not None:
                if st.button("Save Data to Parquet"):
                    save_data_to_parquet(data, "data.parquet")  # Memanggil fungsi save_data_to_parquet
        
        with st.expander("File Info", expanded=True):
            st.write("Production sequence is updated production forecasting data available every week through the application. This data is presented in an Excel file format. It is used not only as a guide for managing inventory throughout the production process but also provides visualization of production data, which will offer valuable insights for decision-making.")

    with from_tab2:
        
        data = None  # Set the data to None if "Sample Data" is chosen again
        parquet_file = "data.parquet"

        if parquet_file is not None:
            data = pd.read_parquet(parquet_file)  # Membaca file Parquet jika diunggah
    
            if data is not None:  # Check if data is not None before using it

                st.write("Total rows in uploaded data:", len(data))
                # Mengambil baris terakhir dari data yang tidak kosong dan memiliki kolom 'VIN' yang bukan "nan"
                
                

                header1, header2 = st.columns([2, 1])
                with header1:
                    last_valid_vin_row = data[data['VIN'] != "nan"].iloc[-1]

                    if not last_valid_vin_row.empty:
                        last_valid_w_on_date = last_valid_vin_row['W-on Date']
                        formatted_w_on_date = last_valid_w_on_date.strftime("%B %d, %Y")

                        info_text = f"Last valid VIN: {last_valid_vin_row['VIN']} on {formatted_w_on_date}"

                        st.info(info_text)

                    else:
                        st.write("Tidak ada VIN yang valid dalam data.")
                with header2:
                    with st.expander("Last Updated Date", expanded=False):
                        if not data.empty:
                            first_row = data.iloc[0]  # Assuming the first row contains the date
                            last_row = data.iloc[-1]  # Get the last row
                            formatted_date_first = first_row["W-on Date"].strftime("%B %d, %Y")
                            formatted_date_last = last_row["W-on Date"].strftime("%B %d, %Y")
                            st.write("First row - W-on Date:", f"<span style='color:red'><b>{formatted_date_first}</b></span>", unsafe_allow_html=True)
                            st.write("Last row - W-on Date:", f"<span style='color:red'><b>{formatted_date_last}</b></span>", unsafe_allow_html=True)

                
                
                # Filter data for a specific year (e.g., 2023) and calculate monthly frequency
                year_to_filter = 2023  # Change this to the desired year
                yearly_counts = filter_data_yearly(data)

                # Create a select box to choose "Model Series"
                selected_model_series = st.selectbox("Select Model Series", yearly_counts['Model Series'].unique(), key="selectbox_3")

                # Periksa apakah pemetaan bulan sudah dilakukan sebelumnya
                if not mapping_done:
                    yearly_counts['Month'] = yearly_counts['Month'].map(month_names)
                    mapping_done = True  # Atur ke True setelah pemetaan dilakukan

                # Filter data based on selected "Model Series"
                selected_data = yearly_counts[yearly_counts['Model Series'] == selected_model_series]

                # Konversi kamus ke daftar
                category_array = list(month_names.values())
                
                col1, col2 = st.columns([2, 1])

                # Create a Plotly bar chart
                fig = px.bar(selected_data, x='Month Year', y='Counts', labels={'Counts': 'Total Production'}, title=f"Monthly Counts for {selected_model_series}")
                # Dalam bagian plotly, gunakan category_array dalam properti categoryarray
                fig.update_xaxes(type='category', categoryorder='array', categoryarray=category_array)
                # Set plot to autosize to the available width
                fig.update_layout(autosize=True)
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Display the DataFrame in the second column
                    selected_data_filtered = selected_data[['Month Year', 'Counts']]
                    selected_data_filtered = selected_data_filtered.reset_index(drop=True) 
                    st.dataframe(selected_data_filtered, use_container_width=True, height=400)

                # Set default end date to today
                # Initialize today_date with the current date
                today_date = datetime.date.today()
                #today_date =  datetime.datetime.combine(today_date, datetime.time())
                print("todaydate :",today_date)

                # Calculate the start date as 1 month ago
                #one_month_ago = today_date - datetime.timedelta(days=30)

                # Date range selection using date_input with default values
                #start_date = st.date_input("Start Date", today_date)
                #end_date = st.date_input("End Date", today_date)
                
                col3, col4 = st.columns([2, 1])

                with col3:
                    st.write("Select date range:")


                    start_date = st.date_input("Start Date", today_date, key="start_date_input2")
                    #start_date = start_date + datetime.timedelta(days=2)
                    end_date = st.date_input("End Date", today_date, key="end_date_input2")

                    # Convert start_date to datetime.datetime
                    start_date_datetime = datetime.datetime(start_date.year, start_date.month, start_date.day)
                    end_date_datetime = datetime.datetime(end_date.year, end_date.month, end_date.day)
                    # Convert the selected dates to datetime.datetime objects
                    #start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
                    #end_date = datetime.datetime.combine(end_date, datetime.datetime.max.time())
                    # Format start_date
                    start_date_str = start_date_datetime.strftime('%Y%m%d')
                    #print("ini strtime :",start_date_str)
                    end_date_str = end_date_datetime.strftime('%Y%m%d')

                    #target_date = datetime.datetime(2023, 10, 20)
                    # Filter data berdasarkan tanggal
                    #target_date_str = target_date.strftime('%Y%m%d')
                    #print("ini strtime example :",target_date_str)
                    #filter_data = data.loc[data['W-on Date'] == target_date_str]
                    #print("tipe data filter_data :", filter_data.dtypes)
                    #print("filter data date :",filter_data)

                with col4:
                    st.write("Select Model Series:")
                    selected_line = st.selectbox("Select Line", list(model_series_mapping.keys()), key="selectbox_4")
                    

                    if not start_date or not end_date:
                        st.warning("Please select the date range.")
                    else:
                        # Use st.session_state to store the selected line
                        if "selected_line" not in st.session_state:
                            st.session_state.selected_line = selected_line
                        else:
                            st.session_state.selected_line = selected_line

                        # Define default values based on the selected line
                        default_model_series = model_series_mapping.get(st.session_state.selected_line, [])
                        valid_default_model_series = [value for value in default_model_series if value in model_series_mapping.get(selected_line, [])]
                        selected_model_series = st.multiselect("Select Model Series", model_series_mapping.get(selected_line, []), default=valid_default_model_series, key="multiselect2")
                        filtered_data = filter_data_date(data, start_date_str, end_date_str, selected_line, selected_model_series)
                        
                        columns_to_display = ['Model Series', 'Model Name', 'MY', 'Country', 'Dom./Exp.', 'W-on Date', 'W-on Shift', 'W-on SEQ']
                        filtered_data_display = filtered_data[columns_to_display]
                        #print(filtered_data_display)

                col5, col6 = st.columns([2, 1])

                with col6:
                    colOpt, colModel = st.columns([1, 1])
                    with colOpt:
                        shift_option = st.radio("Select Shift:", ("All Shifts", "1", "2"), key = "radio2")
                        filtered_data_display = filtered_data_display.reset_index(drop=True)    
                        # Filter data berdasarkan selected_shift
                        if shift_option != "All Shifts":
                            filtered_data_display = filtered_data_display[filtered_data_display['W-on Shift'] == shift_option]

                    with colModel:
                        model_series_counts = filtered_data_display["Model Series"].value_counts()
                        # Create a DataFrame to hold the counts
                        counts_df = pd.DataFrame({"Counts": model_series_counts})
                        # Add a row at the bottom with the total counts
                        counts_df.loc["Total"] = counts_df.sum()

                        model_name_counts = filtered_data_display["Model Name"].value_counts()
                        # Create a DataFrame for "Model Name" counts
                        model_name_counts_df = pd.DataFrame(model_name_counts).reset_index()
                        # Add a date column (for example, the current date)
                        model_name_counts_df["Date"] = [datetime.date.today()] * len(model_name_counts_df)

                        # Rename the columns
                        model_name_counts_df.columns = ["Model Name", "Count", "Date"]
                        st.dataframe(counts_df)
                    with st.expander("Details Type", expanded=False):
                        st.dataframe(model_name_counts_df)
                
                with col5:

                    
                    # Hitung jumlah masing-masing nilai dalam kolom "Country"
                    country_counts = filtered_data_display["Country"].value_counts()

                    # Buat data untuk Pie Chart
                    labels = country_counts.index.tolist()
                    values = country_counts.values.tolist()

                    # Buat Pie Chart dengan innerRadius
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])

                    # Atur layout untuk tampilan lebih baik
                    fig.update_layout(title_text="Production by country")

                    # Tampilkan Pie Chart dalam Streamlit
                    st.plotly_chart(fig, use_container_width=True)

    return data # Return the 'data' variable and the chosen upload option



def save_data_to_parquet(data, filename):
    # Mengkonversi DataFrame ke tabel Arrow
    table = pa.Table.from_pandas(data)
    
    # Menyimpan tabel Arrow dalam format Parquet
    pq.write_table(table, filename)
    st.write(f"Data telah disimpan dalam format Parquet ke {filename}")

def main():
    
    ac.render_cta()
    st.title("Upload and Display Data")

    # Tambahkan option box untuk memilih "Upload" atau "Baca Parquet"
    #upload_option = st.radio("Choose an option:", ("Upload File", "Read Parquet"))

    data = uploaded()  # Store the data returned from the 'uploaded' function and the chosen upload option

    

        

if __name__ == "__main__":
    main()
