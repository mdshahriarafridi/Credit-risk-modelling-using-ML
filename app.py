import streamlit as st
import pandas as pd
import joblib


model = joblib.load('extra_trees_credit_model.pkl')
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ['Sex','Housing', 'Saving accounts', 'Checking account']}