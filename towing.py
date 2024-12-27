import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection as skm
import sklearn.preprocessing as skp
from sklearn import ensemble
from sklearn import metrics
import joblib
import xgboost as xgb
import shap


def read_data(path):
    read_path = path
    files_name = os.listdir(read_path)
    data_list = []
    for name in files_name:
        file_path = os.path.join(read_path, name)
        df = pd.read_csv(file_path)
        data = df.iloc[:, 3:26]
        data_list.append(data)  # --
    whole_data = pd.concat(data_list, axis=0, ignore_index=True)
    print("读取的样本个数和特征个数", whole_data.shape)
    return whole_data


def hgboost(train_x, test_x, train_y, test_y):
    model = ensemble.HistGradientBoostingClassifier(random_state=0)
    model.fit(X=train_x, y=train_y)
    joblib.dump(model, "D:\\factor_analysis\\HGBoost_model.m")
    y_forest_predict = model.predict(test_x)
    print("HGBoost预测测试集准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
    print("HGBoost预测测试集结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
    print("HGBoost预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))


def hgboost_shap(train_x, test_x, train_y, test_y):
    # model = ensemble.HistGradientBoostingRegressor(random_state=0)
    # model.fit(X=train_x, y=train_y)
    # joblib.dump(model, "D:\\factor_analysis\\HGBoost_regressor_model.m")
    model = joblib.load("D:\\factor_analysis\\HGBoost_regressor_model.m")
    explainer = shap.Explainer(model)
    shap_values = explainer(train_x[0:50000, :])
    shap.plots.beeswarm(shap_values)


def xgboost(train_x, test_x, train_y, test_y):
    model = xgb.sklearn.XGBClassifier()
    model.fit(X=train_x, y=train_y)
    joblib.dump(model, "D:\\factor_analysis\\XGBoost_model.m")
    y_forest_predict = model.predict(test_x)
    print("XGBoost预测测试集准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
    print("XGBoost预测测试集结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
    print("XGBoost预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))


def rf(train_x, test_x, train_y, test_y):
    model1 = ensemble.RandomForestClassifier(random_state=0)
    model1.fit(X=train_x, y=train_y)
    joblib.dump(model1, "D:\\factor_analysis\\RF_model.m")
    y_forest_predict = model1.predict(test_x)
    print("随机森林预测测试集准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
    print("随机森林预测测试集结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
    print("随机森林预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))


def gbdt(train_x, test_x, train_y, test_y):
    model1 = ensemble.GradientBoostingClassifier()
    model1.fit(X=train_x, y=train_y)
    joblib.dump(model1, "D:\\factor_analysis\\GBDT_model.m")
    y_forest_predict = model1.predict(test_x)
    print("GBDT预测测试集准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
    print("GBDT预测测试集结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
    print("GBDT预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))


def tow_away_analyze(dataset):
    dataset = dataset.copy()
    data_number = dataset.iloc[:, 0:4]
    data_category = dataset.iloc[:, 4:13]
    # data_category.iloc[0:50, :].to_csv("data-category.csv")  # ***************
    print("采用的特征：", data_number.keys().tolist(), data_category.keys().tolist())
    # data_category = data_category.astype(str)
    TA_label = dataset["TOW_AWAY"]
    # ----------------------------------------------------------------
    imputer = SimpleImputer(strategy='constant')
    # 类别特征one-hot编码和缺失值填充

    # 获取类别特征列的名称和索引
    cat_columns = data_category.select_dtypes(include=['object']).columns.tolist()
    cat_index = data_category.columns.get_indexer(cat_columns)

    # one-hot编码类别特征
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(data_category.iloc[:, cat_index])

    # 创建一个包含one-hot编码后特征的新dataframe
    data_category_onehot = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(cat_columns))

    # data_category_onehot.drop(
    #     ['CHP_VEHTYPE_AT_FAULT_1', 'CHP_VEHTYPE_AT_FAULT_2', 'CHP_VEHTYPE_AT_FAULT_3', 'CHP_VEHTYPE_AT_FAULT_4',
    #      'CHP_VEHTYPE_AT_FAULT_7', 'CHP_VEHTYPE_AT_FAULT_8'], axis=1, inplace=True)

    #  填充缺失值
    data_category_onehot_imputed = imputer.fit_transform(data_category_onehot)

    data_category_onehot_imputed = pd.DataFrame(data_category_onehot_imputed, columns=data_category_onehot.columns)

    cleared_data = pd.concat([data_number, data_category_onehot_imputed], axis=1)
    # ----------------------------------------------------------------
    new_data = pd.concat([cleared_data, TA_label], axis=1)
    new_data = new_data.dropna()

    new_data = new_data.replace({"TOW_AWAY": {'Y': 1, 'N': 0}})  # ---------------
    new_data.iloc[0:500, :].to_csv("sample_.csv")

    np.random.seed(0)
    min_max = skp.MinMaxScaler()
    print("数据量：", new_data.shape)
    X = min_max.fit_transform(new_data.iloc[:, :-1])
    column_names = [column for column in X]
    y = new_data["TOW_AWAY"].values

    train_x, test_x, train_y, test_y = skm.train_test_split(X, y, test_size=0.3, stratify=y)
    print("训练集样本数：", train_x.shape[0])
    print("测试集样本数：", test_x.shape[0])
    # hgboost(train_x, test_x, train_y, test_y)
    # hgboost_shap(train_x, test_x, train_y, test_y)
    # xgboost(train_x, test_x, train_y, test_y)
    # rf(train_x, test_x, train_y, test_y)
    # gbdt(train_x, test_x, train_y, test_y)


def model_train():
    train_model_data = read_data("D:\\factor_analysis\\constructed_data")
    tow_away_analyze(train_model_data)


def model_test():
    test_model_data = read_data("D:\\factor_analysis\\constructed_test_data")

    dataset = test_model_data.copy()
    data_number = dataset.iloc[:, 0:4]
    data_category = dataset.iloc[:, 4:13]
    print("采用的特征：", data_number.keys().tolist(), data_category.keys().tolist())
    data_category = data_category.astype(str)
    TA_label = dataset["TOW_AWAY"]  # 拖运标签
    # ----------------------------------------------------------------
    imputer = SimpleImputer(strategy='constant')
    # 类别特征one-hot编码和缺失值填充

    # 获取类别特征列的名称和索引
    cat_columns = data_category.select_dtypes(include=['object']).columns.tolist()
    cat_index = data_category.columns.get_indexer(cat_columns)

    # one-hot编码类别特征
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoded = onehot_encoder.fit_transform(data_category.iloc[:, cat_index])

    # 创建一个包含one-hot编码后特征的新dataframe
    data_category_onehot = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(cat_columns))

    #  填充缺失值
    data_category_onehot_imputed = imputer.fit_transform(data_category_onehot)

    data_category_onehot_imputed = pd.DataFrame(data_category_onehot_imputed, columns=data_category_onehot.columns)

    cleared_data = pd.concat([data_number, data_category_onehot_imputed], axis=1)
    # ----------------------------------------------------------------
    new_data = pd.concat([cleared_data, TA_label], axis=1)
    new_data = new_data.dropna()
    new_data = new_data.replace({"TOW_AWAY": {'Y': 1, 'N': 0}})  # ---------------

    np.random.seed(0)
    min_max = skp.MinMaxScaler()
    print("数据量：", new_data.shape)
    test_x = min_max.fit_transform(new_data.iloc[:, :-1])
    test_y = new_data["TOW_AWAY"].values

    def hgboost_test(test_x, test_y):
        model = joblib.load("D:\\factor_analysis\\HGBoost_model.m")
        y_forest_predict = model.predict(test_x)
        print("HGBoost预测准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
        print("HGBoost预测结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
        print("HGBoost预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))

    def xgboost_test(test_x, test_y):
        model = joblib.load("D:\\factor_analysis\\XGBoost_model.m")
        y_forest_predict = model.predict(test_x)
        print("XGBoost预测准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
        print("XGBoost预测结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
        print("XGBoost预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))

    def rf_test(test_x, test_y):
        model = joblib.load("D:\\factor_analysis\\RF_model.m")
        y_forest_predict = model.predict(test_x)
        print("随机森林预测准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
        print("随机森林预测结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
        print("随机森林预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))

    def gbdt_test(test_x, test_y):
        model = joblib.load("D:\\factor_analysis\\GBDT_model.m")
        y_forest_predict = model.predict(test_x)
        print("GBDT预测准确率为：", metrics.accuracy_score(test_y, y_forest_predict))
        print("GBDT预测结果与实际结果的混淆矩阵为：\n", metrics.confusion_matrix(test_y, y_forest_predict))
        print("GBDT预测结果评估报告为：\n", metrics.classification_report(test_y, y_forest_predict))

    # hgboost_test(test_x, test_y)
    # xgboost_test(test_x, test_y)
    # rf_test(test_x, test_y)
    # gbdt_test(test_x, test_y)


if __name__ == "__main__":
    model_train()
    # model_test()
