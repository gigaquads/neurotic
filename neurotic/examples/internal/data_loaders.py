import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .paths import TITANIC_CSV_FILEPATH


def titanic_train_test_split(
    filepath=TITANIC_CSV_FILEPATH,
    test_size=0.2,
    one_hot=True,
) -> DataFrame:
    """
    """
    df = pd.read_csv(filepath)

    # remove columns not of interest
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # fill NaN values
    df = df.fillna({
        'Age': df.Age.median(),
        'Embarked': df.Embarked.mode()[0]
    })

    # one-hot encode these columns:
    if one_hot:
        df = pd.get_dummies(df, columns=[
            'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'
        ])

    # scale columns to values between (0, 1)
    scaler = MinMaxScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # x is the df numpy matrix without "Survived" column
    x = df.drop('Survived', axis=1).values
    y = df[['Survived']].values

    # split data into train/test sets
    retval = [df]
    retval.extend(
        train_test_split(x, y, test_size=test_size, random_state=0)
    )
    return retval
