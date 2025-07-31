import streamlit as st
import pandas as pd
import joblib

# Load both models
flight_model = joblib.load("flight_price_pipeline.pkl")
satisfaction_model = joblib.load("customer_satisfaction_pipeline.pkl")

# Sidebar navigation
page = st.sidebar.selectbox("Choose Prediction Type", ["Flight Price", "Customer Satisfaction"])

# ---------------------- FLIGHT PRICE PAGE ----------------------
if page == "Flight Price":
    st.title("✈️ Flight Price Prediction")

    airline = st.selectbox("Airline", ["IndiGo", "Air India", "Jet Airways", "SpiceJet", 
                                       "Vistara", "GoAir", "Multiple carriers", 
                                       "Air Asia", "Trujet", "Multiple carriers Premium economy", 
                                       "Jet Airways Business"])
    source = st.selectbox("Source", ["Delhi", "Kolkata", "Mumbai", "Chennai"])
    destination = st.selectbox("Destination", ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata"])
    total_stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])

    journey_day = st.slider("Journey Day", 1, 31, 15)
    journey_month = st.slider("Journey Month", 1, 12, 5)

    dep_hour = st.slider("Departure Hour", 0, 23, 10)
    dep_min = st.slider("Departure Minute", 0, 59, 30)

    arr_hour = st.slider("Arrival Hour", 0, 23, 12)
    arr_min = st.slider("Arrival Minute", 0, 59, 45)

    duration_hours = st.slider("Duration (in hours)", 0, 50, 2)

    if st.button("Predict Flight Price"):
        input_df = pd.DataFrame({
            "Airline": [airline],
            "Source": [source],
            "Destination": [destination],
            "Total_Stops": [total_stops],
            "Journey_Day": [journey_day],
            "Journey_Month": [journey_month],
            "Dep_hour": [dep_hour],
            "Dep_min": [dep_min],
            "Arr_hour": [arr_hour],
            "Arr_min": [arr_min],
            "Duration_hours": [duration_hours]
        })
        prediction = flight_model.predict(input_df)[0]
        st.success(f"Predicted Price: ₹{int(prediction):,}")

# ------------------ CUSTOMER SATISFACTION PAGE ------------------
elif page == "Customer Satisfaction":
    st.title("😊 Passenger Satisfaction Prediction")

    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    age = st.slider("Age", 0, 100, 30)
    distance = st.slider("Flight Distance", 100, 5000, 500)

    # Ratings
    wifi = st.slider("Inflight Wifi Service", 0, 5, 3)
    time_conv = st.slider("Arrival/Departure Time Convenient", 0, 5, 3)
    booking = st.slider("Ease of Online Booking", 0, 5, 3)
    gate = st.slider("Gate Location", 0, 5, 3)
    food = st.slider("Food and Drink", 0, 5, 3)
    boarding = st.slider("Online Boarding", 0, 5, 3)
    seat = st.slider("Seat Comfort", 0, 5, 3)
    entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
    service = st.slider("On-board Service", 0, 5, 3)
    legroom = st.slider("Leg Room Service", 0, 5, 3)
    baggage = st.slider("Baggage Handling", 0, 5, 3)
    checkin = st.slider("Check-in Service", 0, 5, 3)
    inflight = st.slider("Inflight Service", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 3)
    delay = st.slider("Arrival Delay (in minutes)", 0, 500, 5)

    if st.button("Predict Satisfaction"):
        input_df = pd.DataFrame({
            "Customer_Type": [customer_type],
            "Type_of_Travel": [travel_type],
            "Class": [travel_class],
            "Age": [age],
            "Flight_Distance": [distance],
            "Inflight_wifi_service": [wifi],
            "Departure/Arrival_time_convenient": [time_conv],
            "Ease_of_Online_booking": [booking],
            "Gate_location": [gate],
            "Food_and_drink": [food],
            "Online_boarding": [boarding],
            "Seat_comfort": [seat],
            "Inflight_entertainment": [entertainment],
            "On-board_service": [service],
            "Leg_room_service": [legroom],
            "Baggage_handling": [baggage],
            "Checkin_service": [checkin],
            "Inflight_service": [inflight],
            "Cleanliness": [cleanliness],
            "Arrival_Delay_in_Minutes": [delay]
        })
        prediction = satisfaction_model.predict(input_df)[0]
        st.success(f"Passenger Satisfaction Prediction: {'Satisfied' if prediction == 1 else 'Not Satisfied'}")
