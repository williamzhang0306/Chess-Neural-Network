# Script used to train and export models. Includes option to train locally or from the cloud.
# See cloud-training/Instructions.txt to find commands to run this script.

from keras import layers, models, losses, regularizers, initializers, optimizers, callbacks
from TrainPkg.MyDataHandler import DataGenerator
import pandas as pd
import os


def main(cloud = True, model_type = 'CNN', model_name = 'Model_Cloud12_01_01_23'):

    ### read, parition and load data generators
    if cloud:  
        data = pd.read_csv("gs://chess-ai-bucket/keras-job-dir/chessData.csv")
    else:
        data = pd.read_csv("/Users/williamzhang/Documents/College/Neural-Network-Chess/data/chessData.csv")
    
    num_samples = len(data)
    partition_index = int( num_samples * 0.8 )

    partitioned_IDs = {
        'training' : [i for i in range(0, partition_index)],
        'testing' : [i for i in range(partition_index, num_samples)]
    }

    params = {
        'data_frame' : data,
        'batch_size' : 256, 
        'x_dim' : (8,8,12),
        'y_dim' : (1,1),
    }

    training_data = DataGenerator(IDs = partitioned_IDs['training'], **params)
    testing_data = DataGenerator( IDs = partitioned_IDs['testing'], **params)

    ### Create Model
    model = models.Sequential()

    if model_type == 'MLP':
        model.add(layers.InputLayer( input_shape = (8,8,12)))
        model.add(layers.Flatten())
        model.add(layers.Dense(768, activation = 'elu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(768,activation= 'elu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(350,activation= 'elu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(175,activation= 'elu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(50,activation= 'elu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1,activation='sigmoid'))

    if model_type == 'CNN':
        model.add(layers.Conv2D(filters=20, kernel_size=(5,5), activation='elu',data_format='channels_last', padding = 'same',input_shape = (8,8,12)))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=50, kernel_size=(3,3), activation='elu',data_format='channels_last', padding = 'same'))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(units = 500, activation = 'linear'))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(units = 1, activation= 'sigmoid'))

    model.summary()

    ### Compile and Train Model
    model.compile(
        optimizer="adam",
        loss = losses.mean_squared_error,
        metrics = ['mean_absolute_error']
    )

    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)

    history = model.fit(training_data, 
                        epochs=200, 
                        validation_data=(testing_data), 
                        workers = 6,
                        use_multiprocessing=True,
                        max_queue_size = 100,
                        callbacks= [es],
                        verbose = 2)

    if cloud:
        export_path = os.path.join('gs://chess-ai-bucket/keras-job-dir', 'keras_export')
    else:
        export_path = os.path.join('/Users/williamzhang/Documents/College/Neural-Network-Chess/keras-exports')

    model.save(export_path + model_name)
    print('Model exported to: {}'.format(export_path))


if __name__ == "__main__":
    main(cloud = True, model_type='MLP')