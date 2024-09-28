import mlflow
logged_model = 'runs:/e58f137d7f4647ffa12102b0852089e4/RandomForestClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")