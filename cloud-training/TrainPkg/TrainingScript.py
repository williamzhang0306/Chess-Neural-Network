from keras import layers, models, losses, regularizers, initializers, optimizers, callbacks
from TrainPkg.MyDataHandler import DataGenerator
import pandas as pd
import os

### read, parition and load data generators
def main():

    data = pd.read_csv("gs://chess-ai-bucket/keras-job-dir/random_evals.csv")
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
        'output_type' : 'normalized'
    }

    training_data = DataGenerator(IDs = partitioned_IDs['training'], **params)
    testing_data = DataGenerator( IDs = partitioned_IDs['testing'], **params)

    ### Create Model
    model = models.Sequential()
    model.add(layers.InputLayer( input_shape = (8,8,12)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation = 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2048,activation= 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2048,activation= 'elu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.summary()

    # Compile and Train Model
    # model.compile(
    #     optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.90, beta_2=0.99, epsilon= 1.91828183), 
    #     loss = losses.mean_squared_error, 
    #     metrics=['mean_absolute_error'])
    model.compile(
        optimizer="adam",
        loss = losses.mean_squared_error,
        metrics = ['mean_absolute_error']
    )

    es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 5)

    history = model.fit(training_data, 
                        epochs=200, 
                        validation_data=(testing_data), 
                        workers = 6,
                        use_multiprocessing=True,
                        max_queue_size = 100,
                        callbacks= [es],
                        verbose = 2)

    #model.save('Model_Cloud2_12_19_22')

    export_path = os.path.join('gs://chess-ai-bucket/keras-job-dir', 'keras_export')
    model.save(export_path+"/Model_Cloud5_12_20_22")
    print('Model exported to: {}'.format(export_path))


if __name__ == "__main__":
    main()