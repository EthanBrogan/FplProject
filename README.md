# FplProject
This project uses data science and machine learning to help Fantasy Premier League (FPL) players build better teams and make smarter decisions. It leverages a custom-trained neural network to predict each player's point totals for the upcoming season based on historical and expected performance data.

## Data Source
This project uses the [Fantasy Premier League Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League) by Vaastav Anand. The dataset provides player-level statistics from past Premier League seasons and is maintained by the FPL community.

# Features
Data Collection & Preprocessing: Aggregated season-long stats from a trusted community-maintained FPL dataset (2016/17 to 2024/25). Data includes traditional stats (goals, assists, clean sheets), FPL-specific stats (ICT Index, BPS), and advanced metrics (xG, xA).

Neural Network: A refined multilayer perceptron (MLP) trained to predict total FPL points per player. Achieved a mean absolute error (MAE) of ~1.95 on validation data.

Squad Builder: Generates an optimal 15-player squad (2 GKs, 5 DEFs, 5 MIDs, 3 FWDs) based on predicted points while adhering to FPL rules and budget constraints.

Streamlit Interface:

Recommendations Tab: Suggests players by position or completes a team based on user selections and remaining budget.

Stats Tab: Displays real-time player stats from the FPL API.

Transfers Tab: Shows current transfer trends and pricing data.

## Motivation
Frustrated by the inconsistency of intuition-based team selection, this project aims to democratise access to machine learning tools for casual FPL players. It combines sports analytics with a strong technical foundation in neural networks, data preprocessing, and web-based UI design.

##Future Plans
Improve model performance with multi-season and live-season data.

Deploy online for broader access.

Create and maintain my own data set to improve accuracy.


# Instructions
## 1. clone repo. 
Do this by going to the repo on github (https://github.com/EthanBrogan/FplProject)
Go to the code section and copy the url.


## 2. Requirements
'pip install -r requirements.txt'

## 3. To run the model
'python fpl_model_2425.py'
'python fpl_model_2324.py'

## 4 To run UI
'python -m streamlit run UI.py'

### models
data for 24/25 season trained on 2023/24
data for 23/24 season trained on 2022/23