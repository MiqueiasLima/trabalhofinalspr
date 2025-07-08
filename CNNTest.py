import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization, Activation, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carregamento dos dados
x_train = np.load('CNN/x_train.npy')
y_train = np.load('CNN/y_train.npy')
x_test = np.load('CNN/x_test.npy')
y_test = np.load('CNN/y_test.npy')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)


def create_cnn_model_with_dropout(input_shape):
    """
    Cria um modelo CNN com conexões residuais e Dropout para processamento de sequências.
    """
    inputs = Input(shape=input_shape)

    # Primeira camada convolucional
    x = Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)  # Dropout inicial

    # Bloco residual 1
    residual = x
    x = Conv1D(64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)  # Dropout após a primeira ativação no bloco
    x = Conv1D(64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)  # Dropout após a ativação final do bloco

    # Bloco residual 2
    residual = Conv1D(128, kernel_size=1, padding='same')(x)
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)  # Aumentando ligeiramente o dropout para camadas mais profundas
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # Bloco residual 3
    residual = x
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    # Camada de saída
    x = Dropout(0.4)(x)  # Dropout antes da camada final de convolução
    x = Conv1D(1, kernel_size=1, padding='same')(x)
    outputs = Reshape((input_shape[0],))(x)

    model = Model(inputs, outputs)
    return model


def create_cnn_model(input_shape):
    """
    Cria um modelo CNN com conexões residuais para processamento de sequências.
    """
    inputs = Input(shape=input_shape)

    # Primeira camada convolucional
    x = Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Bloco residual 1
    residual = x
    x = Conv1D(64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])  # Skip connection
    x = Activation('relu')(x)

    # Bloco residual 2
    residual = Conv1D(128, kernel_size=1, padding='same')(x)  # Projeção da dimensão
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])  # Skip connection
    x = Activation('relu')(x)

    # Bloco residual 3
    residual = x
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])  # Skip connection
    x = Activation('relu')(x)

    # Camada de saída
    x = Conv1D(1, kernel_size=1, padding='same')(x)
    outputs = Reshape((input_shape[0],))(x)  # Reshape para (1949,)

    model = Model(inputs, outputs)
    return model


def create_encoder_decoder_model(input_shape):
    """
    Cria um modelo CNN com estrutura de encoder-decoder.
    """
    model = Sequential([
        # Encoder
        Conv1D(64, kernel_size=7, padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),

        Conv1D(128, kernel_size=5, padding='same', strides=2),  # Reduz a dimensão pela metade
        BatchNormalization(),
        Activation('relu'),

        Conv1D(256, kernel_size=3, padding='same', strides=2),  # Reduz a dimensão pela metade novamente
        BatchNormalization(),
        Activation('relu'),

        # Parte central - processamento na resolução mais baixa
        Conv1D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        # Decoder - UpSampling para restaurar a dimensão original
        Conv1D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling1D(size=2),  # Aumenta a dimensão

        Conv1D(128, kernel_size=5, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling1D(size=2),  # Aumenta a dimensão

        Conv1D(64, kernel_size=7, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        # Camada de saída
        Conv1D(1, kernel_size=1, padding='same'),

        # Reshape para obter a saída no formato desejado (1949,)
        Reshape((input_shape[0],))
    ])

    return model


# Criar o modelo
input_shape = (1949, 1)

# Escolha um dos modelos (descomente a linha correspondente ao modelo desejado)
model = create_cnn_model_with_dropout(input_shape)  # Modelo com skip connections
# model = create_encoder_decoder_model(input_shape)  # Modelo encoder-decoder

# Compilar o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Resumo do modelo
model.summary()

# Definir callbacks para melhorar o treinamento
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath='best_model_val_mae.h5',
    monitor='val_mae',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Treinar o modelo
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Avaliar o modelo com o conjunto de teste
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=1)
print(f"Perda no teste: {test_loss}")
print(f"MAE no teste: {test_mae}")

# Plotar a curva de aprendizado
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Curva de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Treino')
plt.plot(history.history['val_mae'], label='Validação')
plt.title('Erro Médio Absoluto (MAE)')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Fazer previsões em alguns exemplos do conjunto de teste
n_samples = 3
predictions = model.predict(x_test[:n_samples])

# Visualizar as previsões comparadas com os valores reais
plt.figure(figsize=(15, 10))
for i in range(n_samples):
    plt.subplot(n_samples, 1, i + 1)
    plt.plot(y_test[i], label='Real', alpha=0.7)
    plt.plot(predictions[i], label='Previsão', alpha=0.7)
    plt.title(f'Amostra {i + 1}')
    plt.legend()

plt.tight_layout()
plt.show()

# Salvar o modelo treinado
model.save('cnn_sequence_model.h5')
print("Modelo salvo como 'cnn_sequence_model.h5'")
