import keras
from keras import layers
from keras import optimizers

#kernel size
filters =5
kernel_size =3

convolution_1d_layer = keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='same', input_shape=(2400, 1), activation="relu", name="convolution_1d_layer")
convolution_1d_2layer = keras.layers.convolutional.Conv1D(7,5, strides=1, padding='same',  activation="relu", name="convolution_1d_2layer")
# 定义最大化池化层
max_pooling_layer = keras.layers.MaxPooling1D(pool_size=2, strides=2,padding="same", name="max_pooling_layer")

# reshape layer
reshape_layer = keras.layers.core.Flatten(name="reshape_layer")

# dropout layer
dropout_layer=keras.layers.Dropout(0.5,name="dropout_layer")

# full connect layer
full_connect_layer1 = keras.layers.Dense(1024, activation="relu", name="full_connect_layer1")
full_connect_layer2 = keras.layers.Dense(512, activation="relu", name="full_connect_layer2")
full_connect_layer3 = keras.layers.Dense(256, activation="relu", name="full_connect_layer3")

model = keras.Sequential()
model.add(convolution_1d_layer)
model.add(convolution_1d_2layer)
model.add(max_pooling_layer)
model.add(reshape_layer)
model.add(dropout_layer)
model.add(full_connect_layer1)
model.add(full_connect_layer2)
model.add(full_connect_layer3)
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
              optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )
model.save('RunCnn.model')
print(model.summary())
