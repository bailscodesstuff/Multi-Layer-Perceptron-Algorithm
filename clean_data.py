import pandas as pd
from sklearn.model_selection import train_test_split


# creating a pandas dataframe from the original data provided
def read_data():
    base_data = pd.read_csv('CWData.csv')

    # columns of the table
    df_flood_data = base_data.loc[:,
                    ['AREA', 'BFIHOST', 'FARL', 'FPEXT', 'LDP', 'PROPWET', 'RMED-1D', 'SAAR', 'Index flood']]

    return df_flood_data


# code to clean data, I will delete rows containing strings or -ve values.
def clean_data(df_flood_data):

    # this loop ensures all values in dataframe are numeric.
    # if not, "coerce" sets the invalid data to NaN
    for col in df_flood_data.columns:
        df_flood_data[col] = pd.to_numeric(df_flood_data[col], errors='coerce')

    # drops NaN values from dataframe
    df_flood_data.dropna(axis=0, inplace=True)
    # ensures all values are positive
    df_flood_data = df_flood_data[(df_flood_data > 0).all(axis=1)]
    # converts dataFrame to csv file
    df_flood_data.to_csv("clean_data.csv", index=False)



def standardise_data(df_flood_data):
    # dictionary to find the max and minimum values for each column
    col_dict = {}
    for col in df_flood_data.columns:
        col_dict[col] = [df_flood_data[col].max(), df_flood_data[col].min()]

    # standardising each value in dataframe and appending to dictionary
    standardised_dict = {}
    # loops through each column in dataframe
    for col in df_flood_data:
        column_data = df_flood_data[col]
        standardised_dict[col] = []
        # loops through each value in the column
        # run the standardisation formula on each value in the column
        for val in column_data:
            s = 0.8 * ((val - col_dict[col][1]) / (col_dict[col][0] - col_dict[col][1])) + 0.1
            standardised_dict[col].append(s)

    # create a new dataframe of standardised data
    standardised_df = pd.DataFrame.from_dict(standardised_dict)
    # writing the dataframe to csv file
    standardised_df.to_csv("standardised_data.csv")
    return standardised_df


def split_data(standardised_df):
    # extract first 20 % of values, append them to new dataframe
    no_rows = len(standardised_df.index)
    no_evaluation_rows = int((no_rows / 100) * 20)

    evaluation_df = pd.DataFrame()
    for i in range(no_evaluation_rows):
        evaluation_df = evaluation_df.append(standardised_df.iloc[i])

    evaluation_df.to_csv("validation_data.csv")

    # 118 represents the rows AFTER the ones that are contained in the evaluation_data file
    test_and_train_df = standardised_df.iloc[118:]

    # randomly splits the data into testing and training
    X = test_and_train_df.drop(['Index flood'], axis=1)
    y = test_and_train_df['Index flood']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    # csv files containing data
    X_train.to_csv("x_train.csv")
    X_test.to_csv("X_test.csv")
    y_train.to_csv("y_train.csv")
    y_test.to_csv("y_test.csv")
    return X_train, X_test, y_train, y_test





df = pd.read_csv("clean_data.csv")
df_validation = pd.read_csv("validaton stuff/validation_data.csv")

standardised_df = standardise_data(df)
X_train, X_test, y_train, y_test = split_data(standardised_df)
