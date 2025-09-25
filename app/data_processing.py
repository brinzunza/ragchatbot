import pandas as pd

def process_data():
    df = pd.read_csv("../testing_tools/data_analysis/data.csv")
    df.drop(columns=['Address'], inplace=True)
    object_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=object_cols)
    df_encoded = df_encoded.fillna(0)

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)

    output_file_path = 'files/clean_data.csv'
    df_encoded.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    process_data()