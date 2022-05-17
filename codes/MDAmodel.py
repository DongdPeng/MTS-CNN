from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, concatenate

def build_MDA(input_dims, encoding_dims):
    """
    Function for building multimodal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(int(encoding_dims[0]/len(input_dims)),
                                   activation='sigmoid')(input_layers[j]))

    # Concatenate layers
    if len(encoding_dims) == 1:
        hidden_layer = concatenate(hidden_layers, name='middle_layer')
    else:
        hidden_layer = concatenate(hidden_layers)

    # middle layers
    for i in range(1, len(encoding_dims)-1):
        if i == int(len(encoding_dims)/2):
            hidden_layer = Dense(encoding_dims[i],
                                 name='middle_layer',
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)

    if len(encoding_dims) != 1:
        # reconstruction of the concatenated layer
        hidden_layer = Dense(encoding_dims[0],
                             activation='sigmoid')(hidden_layer)

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(int(encoding_dims[-1]/len(input_dims)),
                                   activation='sigmoid')(hidden_layer))
    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   activation='sigmoid')(hidden_layers[j]))

    # autoencoder model
    sgd = SGD(lr=0.2, momentum=0.95, decay=1e-6, nesterov=True)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])
    print (model.summary())
    return model

