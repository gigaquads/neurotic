from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from neurotic.data.loaders import SingleStepTimeSeriesDataFrameLoader
from neurotic.training import Trainer

from .internal.data_loaders import jena_climate_2009_2016


def main():
    # Create a RNN using LSTM cells. We will feed it batches of 32 (inputs,
    # labels) pairs each.
    model = Sequential([
        LSTM(32, return_sequences=True),
        Dense(units=1)
    ])

    # create a trainer that "stops early" if the the "val_loss" metric
    # ceases to change after 2 consecutive epochs.
    trainer = Trainer(
        metrics=[MeanAbsoluteError()],
        callbacks=[
            EarlyStopping(
                # See: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
                # patience is the number of epochs with no improvement after
                # which training will be stopped.
                patience=2, 
            ),
            ModelCheckpoint(
                # See: https://www.tensorflow.org/tutorials/keras/save_and_load
                # specify how to save the trained model to disk
                filepath='./models/test/test.ckpt',
                save_weights_only=True,
                verbose=1
            )
        ]
    )

    # create a loader that prepares time series data, stored in a pandas
    # DataFrame, by converting it to batched input and label tensors, as
    # expected by the model. Note that `key` is the name of the feature column
    # we are trying to predict.
    loader = SingleStepTimeSeriesDataFrameLoader(
        key='T (degC)', period=24, step=1, batch_size=32
    )

    # load pre-sanitized dataframes
    df_train, df_val, df_test = jena_climate_2009_2016()

    # convert dataframes to training, validation, testing Datasets
    ds_train, ds_val = loader.load([df_train, df_val])

    # compile and fit the model (for at most 10 epochs)
    trainer.train(model, 10, ds_train, ds_val)

    model.summary()


if __name__ == '__main__':
    exit(main())