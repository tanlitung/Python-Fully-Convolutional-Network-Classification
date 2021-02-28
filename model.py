from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D, Activation
from tensorflow.keras.models import Model


def FCN_model(len_classes=10, dropout_rate=0.2):
    """ Initialize Generator object.
    Args
        len_classes            : Number of classes
        dropout_rate           : Rate of dropout to be used
    """
    input = Input(shape=(None, None, 1))

    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(100, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    x = Conv2D(len_classes, (1, 1))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)
    predictions = Activation('softmax')(x)

    model = Model(inputs=input, outputs=predictions)

    print(model.summary())
    print(f'Total number of layers for FCN: {len(model.layers)}')

    return model