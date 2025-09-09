import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def get_schedule(start_station, end_station, stations, first_departure, last_departure, direction):
    """
    Generates a list of train schedules for a given direction.
    """
    schedules = []
    current_departure = first_departure
    train_number = 1

    while current_departure <= last_departure:
        current_time = current_departure
        
        # Determine the sequence of stations
        if direction == "towards Station 16":
            station_list = stations
        else:
            station_list = stations[::-1]

        for i, station in enumerate(station_list):
            arrival_time = current_time
            
            # The last station is a destination, no departure time from there
            if i == len(station_list) - 1:
                departure_time = None
            else:
                departure_time = arrival_time + timedelta(seconds=30)
                current_time = departure_time + timedelta(minutes=5)
            
            schedules.append({
                'Train Number': train_number,
                'Direction': direction,
                'Station': station,
                'Arrival Time': arrival_time.strftime("%I:%M:%S %p"),
                'Departure Time': departure_time.strftime("%I:%M:%S %p") if departure_time else "Destination"
            })
            
        current_departure += timedelta(minutes=19)
        train_number += 1
        
    return schedules

# Set up the Streamlit page
st.set_page_config(
    page_title="Train Schedule Generator",
    page_icon="🚆"
)

st.title("🚆 Train Schedule Generator")

st.write(
    """
    This application calculates and displays the daily train schedule for a route with 16 stations.
    The schedule is generated based on the following rules:
    - Trains depart every 19 minutes.
    - Travel time between stations is 5 minutes.
    - Station stops are 30 seconds.
    """
)

# User input for date selection
selected_date = st.date_input("Select a date to view the schedule:")
day_of_week = selected_date.weekday() # Monday=0, Sunday=6

# Determine start times based on the day of the week
if day_of_week == 6:  # Sunday
    st.info(f"The schedule for **Sunday** has a first departure at 7:00 AM.")
    first_train_time = datetime.strptime("07:00:00", "%H:%M:%S").time()
else: # Weekdays and Saturday
    st.info(f"The schedule for **{selected_date.strftime('%A')}** has a first departure at 6:30 AM.")
    first_train_time = datetime.strptime("06:30:00", "%H:%M:%S").time()

# Define the full schedules with datetime objects
start_date_time = datetime.combine(selected_date, first_train_time)
last_departure_1 = datetime.combine(selected_date, datetime.strptime("23:55:00", "%H:%M:%S").time())
last_departure_16 = datetime.combine(selected_date, datetime.strptime("22:52:00", "%H:%M:%S").time())

# List of stations
stations = [f"Station {i}" for i in range(1, 17)]

# Generate schedules
schedule_1_to_16 = get_schedule(stations[0], stations[-1], stations, start_date_time, last_departure_1, "towards Station 16")
schedule_16_to_1 = get_schedule(stations[-1], stations[0], stations, start_date_time, last_departure_16, "towards Station 1")

# Create dataframes
df_1_to_16 = pd.DataFrame(schedule_1_to_16)
df_16_to_1 = pd.DataFrame(schedule_16_to_1)

# Display schedules
st.header("Schedule: Station 1 to Station 16")
st.dataframe(df_1_to_16, use_container_width=True)

st.header("Schedule: Station 16 to Station 1")
st.dataframe(df_16_to_1, use_container_width=True)
