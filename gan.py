import numpy as np
import glob
import matplotlib.pyplot as plt

from keras.layers import LSTM, Dense, Reshape, LSTM, BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

from midi_dataframe import MidiDataframe

midi_dataframe = MidiDataframe()
for i, file in enumerate(glob.glob("musicxml/*")):
    print(f'Loading file {file}')
    midi_dataframe.load_midi(file, i, remove_accidental=False)

midi_dataframe.column_to_map(orderby='frequency')
midi_dataframe.scaler_transform()

music_list = midi_dataframe.get_slices_note_length(100)
print(len(music_list))

# Constants
LATENT_DIM = 1000
N_FEATURES = 2
EPOCHS = 150
BATCH_SIZE = 64
SHAPE = (len(music_list[0]), N_FEATURES) # x timesteps, 2 features (note and duration)

# Discriminator
def define_discriminator(shape, optimizer):
    discriminator = Sequential(name='Discriminator')
    discriminator.add(LSTM(128, input_shape=shape, return_sequences=True))
    discriminator.add(LSTM(128, return_sequences=True))
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(LSTM(128, return_sequences=True))
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(LSTM(128, return_sequences=True))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.summary()

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return discriminator

# Generator
def define_generator(latent_dim, shape, optimizer):
    generator = Sequential(name='Generator')
    generator.add(LSTM(128, input_dim=latent_dim, return_sequences=True))
    generator.add(LSTM(128, return_sequences=True))
    generator.add(Dense(64))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LSTM(128, return_sequences=True))
    generator.add(Dense(64))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LSTM(128, return_sequences=True))
    generator.add(Dense(np.prod(shape), activation='tanh'))
    generator.add(Reshape(shape))
    generator.summary()

    generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return generator

def define_gan(generator, discriminator, optimizer):
    discriminator.trainable = False
    
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def train(generator, discriminator, gan, dataset, LATENT_DIM, EPOCHS, BATCH_SIZE):
    for epoch in range(EPOCHS):
        real_notes = dataset[np.random.choice(len(dataset), size=BATCH_SIZE, replace=False)]
        discriminator_real_loss, discriminator_real_accuracy = discriminator.train_on_batch(real_notes, real)

        noise = np.random.rand(BATCH_SIZE, 1, LATENT_DIM)
        fake_notes = generator.predict(noise)
        discriminator_fake_loss, discriminator_fake_accuracy = discriminator.train_on_batch(fake_notes, fake)

        noise = np.random.rand(BATCH_SIZE, 1, LATENT_DIM)
        gan_loss = gan.train_on_batch(noise, real)

        print(f'E={epoch}, DRL={discriminator_real_loss}, DFL={discriminator_fake_loss}, GL={gan_loss}, DRA={discriminator_real_accuracy}, DFA={discriminator_fake_accuracy}')
        
        d_real_loss.append(discriminator_real_loss)
        d_fake_loss.append(discriminator_fake_loss)
        g_loss.append(gan_loss)

        midi_dataframe.load_prediction(fake_notes[:8])
        midi_dataframe.scaler_inverse_transform()
        midi_dataframe.map_to_column()
        midi_dataframe.save_midi('gan', epoch)

optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
dataset = np.array(music_list)
real = np.ones((BATCH_SIZE, SHAPE[0], 1))
fake = np.zeros((BATCH_SIZE, SHAPE[0], 1))

d_real_loss = []
d_fake_loss = []
g_loss = []

discriminator = define_discriminator(SHAPE, optimizer)
generator = define_generator(LATENT_DIM, SHAPE, optimizer)
gan = define_gan(generator, discriminator, optimizer)
train(generator, discriminator, gan, dataset, LATENT_DIM, EPOCHS, BATCH_SIZE)

plt.plot(d_real_loss, c='green')
plt.plot(d_fake_loss, c='red')
plt.plot(g_loss, c='blue')
plt.title("GAN Loss per Epoch")
plt.legend(['d_real', 'd_fake', 'g'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('plots/gan_loss.png')
plt.close()