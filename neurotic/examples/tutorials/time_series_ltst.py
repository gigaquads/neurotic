"""
https://www.tensorflow.org/tutorials/structured_data/time_series
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tabulate

from ..internal.data_loaders import jena_climate_2009_2016


def main():
    df, time, (df_train, df_val, df_test) = jena_climate_2009_2016()

    shift = 1
    input_width = 24
    label_width = 24
    total_window_width = input_width + shift
    input_slice = slice(input_width)
    input_indices = np.arange(total_window_width)[input_slice]
    label_columns = ['T (degC)']
    label_column_indices = {k: i for i, k in enumerate(label_columns)}
    column_indices = {k: i for i, k in enumerate(df.columns)}
    label_start = total_window_width - label_width
    labels_slice = slice(label_start, None)
    label_indices = np.arange(total_window_width)[labels_slice]

    split = lambda x: split_window(
        x, input_width, label_width, column_indices, label_columns,
        input_slice, labels_slice
    )

    ds_train = make_dataset(df_train, split, total_window_width)
    ds_val = make_dataset(df_val, split, total_window_width)

    example = get_example(ds_train)

    print(
        tabulate.tabulate([
            ('Total Window Size', total_window_width),
            ('Input Indices', input_indices),
            ('Label Indices', label_indices),
            ('Label Column Name(s)', label_columns)
            ], tablefmt='fancy_grid'
        ))


    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(model, ds_train, ds_val)

    plot(model, 'T (degC)', example, input_indices, column_indices,
         label_indices, label_columns, label_column_indices)


def compile_and_fit(model, ds_train, ds_val, epochs=5, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=[early_stopping]
    )
    return history


def make_dataset(data, func_split_window, total_window_size):
    return tf.keras.preprocessing.timeseries_dataset_from_array(
        data=np.array(data, dtype=np.float32),
        targets=None,
        sequence_length=total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=32,
    ).map(func_split_window)


def split_window(
    features, input_width, label_width, column_indices,
    label_columns, input_slice, labels_slice
):
    inputs = features[:, input_slice, :]
    labels = features[:, labels_slice, :]
    if label_columns is not None:
        labels = tf.stack(
            [labels[:, :, column_indices[name]] for name in label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, input_width, None])
    labels.set_shape([None, label_width, None])
    return inputs, labels


def get_example(ds_train):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(get_example, '_memoized', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(ds_train))
        # And cache it for next time
        get_example._memoized = result
    return result


def plot(
    model,
    plot_col,
    example,
    input_indices,
    column_indices,
    label_indices,
    label_columns,
    label_columns_indices,
    max_subplots=3
):
    plt.figure(figsize=(12, 8))

    inputs, labels = example
    plot_col_index = column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))

    for n in range(max_n):
        plt.subplot(3, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(input_indices,
                 inputs[n, :, plot_col_index],
                 label='Inputs',
                 marker='.',
                 zorder=-10)

        if label_columns:
            label_col_index = label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(label_indices,
                    labels[n, :, label_col_index],
                    edgecolors='k',
                    label='Labels',
                    c='#2ca02c',
                    s=64)

        if model is not None:
            predictions = model(inputs)
            plt.scatter(label_indices,
                        predictions[n, :, label_col_index],
                        marker='X',
                        edgecolors='k',
                        label='Predictions',
                        c='#ff7f0e',
                        s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')
    plt.show()


if __name__ == '__main__':
    exit(main())