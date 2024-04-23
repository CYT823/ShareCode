import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from imblearn.over_sampling import SMOTE
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb

class Models():
    def __init__(self):
        # load data
        self.data = pd.read_csv("data/training_data_NEW_encode.csv")

        # # 切分model
        # part1_data = data[data['City'].isin([1, 2, 4, 5, 8, 13, 14])]
        # part2_data = data[data['City'].isin([3, 6, 7, 9, 10, 11, 12, 15, 16, 17])]
        # part3_data = data[data['City'].isin([18, 19, 20, 21, 22])]
        # # 重新encode City, Block 

        # drop useless data information
        self.droplist = [
            # "AttachedArea",
            # "Number_of_Bank_Around",
            # "Number_of_PostOffice_Around",
            # "Number_of_ATM_Around",
            # "Number_of_Hospital_Around",
            # "Number_of_ConvienceStore_Around",
            # "Number_of_MRT_Around",
            # "Number_of_Train_Around",
            "Number_of_Bike_Around",
            "Number_of_Bus_Around",
            "Number_of_College_Around",
            "Number_of_HighSchool_Around",
            "Number_of_JuniorHighSchool_Around",
            "Number_of_ElementarySchool_Around",
            # "Is_Bank_Nearby",
            # "Is_PostOffice_Nearby",
            # "Is_ATM_Nearby",
            # "Is_Hospital_Nearby",
            # "Is_ConvienceStore_Nearby",
            # "Is_MRT_Nearby",
            # "Is_Train_Nearby",
            "Is_Bike_Nearby",
            "Is_Bus_Nearby",
            "Is_College_Nearby",
            "Is_HighSchool_Nearby",
            "Is_JuniorHighSchool_Nearby",
            "Is_ElementarySchool_Nearby"]
        self.data.drop(self.droplist, axis=1, inplace=True)

        # specify the features(X) and the target variable(y)
        X = self.data.drop(["Value"], axis=1)
        y = self.data["Value"]
        
        # split the data into training and testing sets (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    def LR_model(self):
        # create a linear regression model
        self.model = LinearRegression()

        # train the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # make predictions on the testing data
        self.y_pred = self.model.predict(self.X_test)

        # evaluate the model's performance
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print("====== Linear Regression ======")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    def PR_model(self):
        degree = 2
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(self.X_train)

        # create a linear regression model
        self.model = LinearRegression()

        # train the model on the training data
        self.model.fit(X_train_poly, self.y_train)

        # make predictions on the testing data
        X_test_poly = poly_features.transform(self.X_test)
        self.y_pred = self.model.predict(X_test_poly)

        # evaluate the model's performance
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print("====== Polynomial Regression ======")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
    def SVM_model(self):
        # create a SVM model
        self.model = svm.SVR()

        # train the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # make predictions on the testing data
        self.y_pred = self.model.predict(self.X_test)

        # evaluate the model's performance
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print("====== SVM ======")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    def RF_model(self):
        # create a Random Forest model
        self.model = RandomForestRegressor(n_estimators=10)

        # train the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # make predictions on the testing data
        self.y_pred = self.model.predict(self.X_test)

        # evaluate the model's performance
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print("====== Random Forest ======")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    def CB_model(self):
        # create a Cat Boost model
        # self.model = CatBoostRegressor()
        self.model = CatBoostRegressor(loss_function="MAPE", iterations=10000, learning_rate=0.1, depth=8, verbose=False)

        # train the model on the training data
        self.model.fit(
            self.X_train, 
            self.y_train, 
            eval_set=Pool(self.X_test, self.y_test), 
            use_best_model=True, 
            plot=True,
            early_stopping_rounds=20,
        )

        # make predictions on the testing data
        self.y_pred = self.model.predict(self.X_test)

        # evaluate the model's performance
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print("====== Cat Boost ======")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    def XGB_model(self):
        # create a XGBoost model
        self.model = xgb.XGBRegressor(
                objective="reg:squarederror",               # Specify regression as the objective
                eval_metric=mean_squared_error,
                early_stopping_rounds=10,
                max_depth=10,                               # Maximum tree depth
                learning_rate=0.005,                        # Learning rate
                # gamma=0.001,                              # Minimum loss reduction required to make a further partition
                n_estimators=2500,                          # Number of boosting rounds
        )
        # train the model on the training data
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])

        # make predictions on the testing data
        self.y_pred = self.model.predict(self.X_test)

        # evaluate the model's performance
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print("====== XGBoost ======")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    def LGBM_model(self):
        # create a XGBoost model
        self.model = lgb.LGBMRegressor(
                        objective="regression",   # Specify regression as the objective
                        metric="mape",           # Evaluation metric (MAPE)
                        boosting_type="gbdt",    # Gradient Boosting Decision Tree
                        num_leaves=60,           # Number of leaves in a tree
                        learning_rate=0.05,      # Learning rate
                        n_estimators=1000         # Number of boosting rounds
                    )

        # train the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # make predictions on the testing data
        self.y_pred = self.model.predict(self.X_test)

        # evaluate the model's performance
        mse = mean_squared_error(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print("====== LightGBM ======")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

    def run_predict(self):
        # load testing/evaluating dataset
        self.prediction_data = pd.read_csv("data/testing_data_NEW_encode.csv")
        self.prediction_data.drop(self.droplist, axis=1, inplace=True)

        # create a DataFrame to store the prediction
        y_pred = self.model.predict(self.prediction_data)
        predictions_df = pd.DataFrame({
            'Predicted_Price': y_pred  
        })

        # save the result to CSV
        output_csv_path = "data/predictions_ML.csv"
        predictions_df.to_csv(output_csv_path, index=False)
        print(f"save file to {output_csv_path}")

    def draw_correlation(self):
        # correlation plot
        plt.figure(figsize=(10, 10))
        sns.heatmap(self.data.corr(),
                    cmap = 'BrBG',
                    fmt = '.2f',
                    linewidths = 2,
                    annot = True)
        plt.show()

if __name__ == "__main__":
    models = Models()
    # models.LR_model()
    # print()
    # models.PR_model()
    # print()
    # models.SVM_model()
    # print()
    # models.RF_model()
    # print()
    # models.CB_model()
    # print()
    models.XGB_model()
    # print()
    # models.LGBM_model()
    # print()
    models.run_predict()


    # 建物面積 包含 主建屋面積 ＋ 附屬建物面積
    # 附屬建物面積 不含 陽台面積
    # 單價不含車位價格


    # 創造新feature:
    # 1.有沒有臨近醫院、學校、便利商店...
    # 計算物件資料與各公共設施距離，以方圓一公里為門檻
    # 2.Amenity Density 密度：
    # 例如：物件2公里內的設施個數
    # 3.依據北中南東劃分模型
