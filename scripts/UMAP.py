from principal_component_analysis import load_in_data
from principal_component_analysis import make_pandas_dataframes



if __name__ == '__main__':
    encoded_linkers = load_in_data() #currently loading in reduced dataset
    df_noscaler, df_stdscaler, df_mmscaler = make_pandas_dataframes(encoded_linkers)
