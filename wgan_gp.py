# Based on https://keras.io/examples/generative/wgan_gp/

import tensorflow as tf
from keras.layers import LSTM, Dense, Reshape, LSTM, LeakyReLU, BatchNormalization, LayerNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import Callback

import numpy as np
import glob
import matplotlib.pyplot as plt

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
N_FEATURES = 2 # Note and Duration
EPOCHS = 150
BATCH_SIZE = 64
DISCRIMINATOR_EXTRA_STEPS = 5
GRADIENT_PENALTY_WEIGHT = 10
SHAPE = (len(music_list[0]), N_FEATURES) # x timesteps, 2 features (note and duration)

# Discriminator
def get_discriminator_model():
    discriminator = Sequential(name='Discriminator')
    discriminator.add(LSTM(128, input_shape=SHAPE, return_sequences=True))
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(LSTM(128, return_sequences=True))
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(LSTM(128, return_sequences=True))
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(LSTM(128, return_sequences=True))
    discriminator.add(Dense(1)) 
    return discriminator

# Generator
def get_generator_model():
    generator = Sequential(name='Generator')
    generator.add(LSTM(128, input_dim=LATENT_DIM, return_sequences=True))
    generator.add(LSTM(128, return_sequences=True))
    generator.add(Dense(64))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(LayerNormalization())
    generator.add(LSTM(128, return_sequences=True))
    generator.add(Dense(64))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(LayerNormalization())
    generator.add(LSTM(128, return_sequences=True))
    generator.add(Dense(np.prod(SHAPE), activation='tanh'))
    generator.add(Reshape(SHAPE))
    return generator

# Discriminator Loss Function
def discriminator_loss(real_notes, fake_notes):
    real_loss = tf.reduce_mean(real_notes)
    fake_loss = tf.reduce_mean(fake_notes)
    return fake_loss - real_loss

# Generator Loss Function
def generator_loss(fake_notes):
    return -tf.reduce_mean(fake_notes)

# WGAN-GP
class WGAN_GP(Model):
    def __init__(self, discriminator, generator, latent_dim, batch_size, discriminator_extra_steps=5, gradient_penalty_weight=10.0):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.discriminator_extra_steps = discriminator_extra_steps
        self.gradient_penalty_weight = gradient_penalty_weight

    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss_function, generator_loss_function):
        super().compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_loss_function = discriminator_loss_function
        self.generator_loss_function = generator_loss_function

    def gradient_penalty(self, real_notes, fake_notes):
        alpha = tf.random.normal([self.batch_size, 1, 1], 0.0, 1.0)
        diff = fake_notes - real_notes
        interpolated = real_notes + alpha * diff

        with tf.GradientTape() as gradient_penalty_tape:
            gradient_penalty_tape.watch(interpolated)
            prediction = self.discriminator(interpolated, training=True)

        grads = gradient_penalty_tape.gradient(prediction, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, real_notes):
        for _ in range(self.discriminator_extra_steps):
            random_latent_vectors = tf.random.normal(shape=(self.batch_size, 1, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_notes = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_notes, training=True)
                real_logits = self.discriminator(real_notes, training=True)

                discriminator_cost = self.discriminator_loss_function(real_notes=real_logits, fake_notes=fake_logits)
                gradient_penalty = self.gradient_penalty(real_notes, fake_notes)
                discriminator_loss = discriminator_cost + gradient_penalty * self.gradient_penalty_weight

            discriminator_gradient = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(self.batch_size, 1, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_notes = self.generator(random_latent_vectors, training=True)
            generated_logits = self.discriminator(generated_notes, training=True)
            generator_loss = self.generator_loss_function(generated_logits)

        gen_gradient = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {"discriminator_loss": discriminator_loss, "generator_loss": generator_loss}

class GANMonitor(Callback):
    def __init__(self, latent_dim, batch_size, length):
        self.batch_size = batch_size
        self.length = length
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, 1, self.latent_dim))
        generated_notes = self.model.generator(random_latent_vectors)

        midi_dataframe.load_prediction(generated_notes[:self.length])
        midi_dataframe.scaler_inverse_transform()
        midi_dataframe.map_to_column()
        midi_dataframe.save_midi('wgan-gp', epoch)

discriminator_model = get_discriminator_model()
discriminator_model.summary()
generator_model = get_generator_model()
generator_model.summary()

# Optimizers
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

# Monitor Keras Callback
gan_callback = GANMonitor(latent_dim=LATENT_DIM, batch_size=BATCH_SIZE, length=8)

# WGAN-GP
wgan = WGAN_GP(discriminator=discriminator_model, generator=generator_model, latent_dim=LATENT_DIM, batch_size=BATCH_SIZE, discriminator_extra_steps=DISCRIMINATOR_EXTRA_STEPS, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
wgan.compile(discriminator_optimizer=discriminator_optimizer, generator_optimizer=generator_optimizer, generator_loss_function=generator_loss, discriminator_loss_function=discriminator_loss)

# Train
dataset = np.array(music_list)
dataset = dataset[:-(dataset.shape[0] % BATCH_SIZE)] # Drop some data to fit the format required
loss = wgan.fit(dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[gan_callback])

plt.plot(loss.history['discriminator_loss'], c='red')
plt.plot(loss.history['generator_loss'], c='blue')
plt.title("WGAN-GP Loss per Epoch")
plt.legend(['d_loss', 'g_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('plots/wgan-gp_loss.png')
plt.close()