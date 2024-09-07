from typing import Callable, Tuple

from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.preprocessing import LabelEncoder

def apply(df: DataFrame, prop: str, convert: Callable[[str], str]) -> Tuple[DataFrame, LabelEncoder]:
    to_prop = convert(prop)

    le = LabelEncoder()
    le.fit(df[prop])

    copied = df.copy()
    copied[to_prop] = le.transform(df[prop])

    return (copied, le)

def replace(df: DataFrame, prop: str, convert: Callable[[str], str]) -> Tuple[DataFrame, LabelEncoder]:
    to_prop = convert(prop)

    le = LabelEncoder()
    le.fit(df[prop])

    df[to_prop] = le.transform(df[prop])

    return (df, le)
