# Ethiopian Vehicle Insurance Premiums

This is a Python data analysis and linear regression / XGBoost modelling project with Pandas and scikit-learn.

The main aim of this project is to **predict** vehicle insurance premiums using insurance policy data such as gender, vehicle manufacturer, claims paid, etc.

The data was sourced from [Mendeley Data](https://data.mendeley.com/datasets/34nfrk36dt/1).

The trained XGBoost model is available as a Docker container. In order to run, follow these steps:

0. Install Docker & Culr (or an alternative).

1. Download the latest release and extract it.

2. Run this code inside the downloaded directory:
```
docker build -t insurance-xgb .
docker run -p 8080:8080 insurance-xgb
```

3. Post a request with curl or an alternative:
```
curl -X POST "http://127.0.0.1:8080/predict" \
    -H "Content-Type: application/json" \
    -d '{"SEX":1,"INSR_TYPE":"PRIVATE","INSURED_VALUE":100000,"INSR_COVER":"LIABILITY","TYPE_VEHICLE":"Truck","PROD_YEAR":2011,"VEHICLE_AGE":5,"WAS_CLAIM_PAID":0,"PREVIOUS_CLAIM_PAID":0,"PREVIOUS_POLICYHOLDERS":2}'
```