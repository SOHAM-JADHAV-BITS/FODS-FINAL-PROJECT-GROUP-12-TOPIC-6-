# FODS-FINAL-PROJECT-GROUP-12-TOPIC-6


##DATA FILES AND LINK FOR SOURCING DATA FILES 
[https://www.kaggle.com/datasets/deepaksirohiwal/delhi-air-quality](https://www.kaggle.com/datasets/deepaksirohiwal/delhi-air-quality)

1.delhi_aqi.csv  

2.delhi_aqi_minmax_normalized.csv  

3.processed_data.csv 

4.FINAL.xlsx



##USE GOOGLE COLLAB TO RUN ALL THE .ipynb FILES
For each of the Google Collab files mentioned , load all the 4 data files mentioned before,  into the environment and install necessary packages and then run the cells.



#FOR THE FINAL DASHBOARD VIA STREAMLIT 
ensure streamlit , joblib , os , and all other necessary packages like xgboost , catboost , pandas , pickle , json , datetime , subprocess , sys , sklearn or scikit-learn
after installing these packages in a venv python environment(ie navigate to a folder venv after creating it on the desktop and bash the pip install commands from the cmd after navigating to the directory) add the files -:
named :
rf_model.pkl
xg_model_booster.json
catboost_model.pkl
scaler_params.json
cat_model.cbm

into the same environment or the folder venv . After following the steps -> use CMD to navigate to the current evironment  or the folder venv and activate it 
for eg>
Open CMD - (in my case i navigated to this directory) 
C:\Users\SOHAM\OneDrive\Desktop\venv> 

then to actovate the environment,type  this -
C:\Users\SOHAM\OneDrive\Desktop\venv>Scripts\activate
hit enter 

then type ->
streamlit run stream1.py 








