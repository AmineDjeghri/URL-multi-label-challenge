from preprocessing import Preprocess




def main():
    parquet_data_path = "../data/"
    preprocess = Preprocess()
    df = preprocess.create_dataframe(parquet_data_path, preprocess=True)

    print(df.columns)

if __name__ == '__main__':
    main()
