import os
import shutil

from typing import Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .paths import TITANIC_CSV_FILEPATH, JENA_CLIMATE_CSV_FILEPATH


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


def jena_climate_2009_2016(split=(0.7, 0.2, 0.1)) -> Tuple:
    if not os.path.exists(JENA_CLIMATE_CSV_FILEPATH):
        data_zip_filename = 'jena_climate_2009_2016.csv.zip'
        data_csv_uri = (
            'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
            + data_zip_filename
        )
        zip_path = tf.keras.utils.get_file(
            origin=data_csv_uri, fname=data_zip_filename, extract=True
        )
        csv_filepath, _ = os.path.splitext(zip_path)
        shutil.copyfile(csv_filepath, JENA_CLIMATE_CSV_FILEPATH)
        df = pd.read_csv(csv_filepath)
    else:
        df = pd.read_csv(JENA_CLIMATE_CSV_FILEPATH)

    time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    time_s = time.map(datetime.timestamp)

    # sanitize bad data...
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # angles don't make good input. For example, 360 and 0 represent the same
    # direction. Convert from polar to rectangular coordinates.
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)')*np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    # apply simple formula to extract the seasonality signal
    # as sine waves.
    day = 24 * 60 * 60
    year = 365.2425 * day

    df['Day sin'] = np.sin(time_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(time_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(time_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(time_s * (2 * np.pi / year))

    # partition df into train, validation, and test sets
    n = len(df)
    i = int(n * split[0])
    j = i + int(n * split[1])
    k = j + int(n * split[2])
    x_train = df[0:i]
    x_validation = df[i:j]
    x_test = df[j:k]

    train_mean = x_train.mean()
    train_std = x_train.std()

    # normalize them
    x_train = (x_train - train_mean) / train_std
    x_validation = (x_validation - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    return (x_train, x_validation, x_test)