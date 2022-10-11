

def remove_chars_space(df, col_name):
    chars = ["\'", ",", "]", "[", " "]
    for char in chars:
        df[col_name] = df[col_name].str.replace(char, "", regex=False)
    return df


def get_dct_from_df(df):
    """works only with two columns"""
    keys = df[df.columns[0]].tolist()
    values = df[df.columns[1]].tolist()
    return dict(zip(keys, values))