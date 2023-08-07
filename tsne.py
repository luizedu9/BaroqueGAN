import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from midi_dataframe import MidiDataframe

# Real Data
midi_dataframe = MidiDataframe()
for i, file in enumerate(glob.glob("musicxml/*")):
    print(f'Loading file {file}')
    midi_dataframe.load_midi(file, i)
music_list = midi_dataframe.get_slices_note_length(100)
dataset = np.array(music_list)
real_data = dataset.reshape(dataset.shape[0], dataset.shape[1]*dataset.shape[2])

# GAN Generated Data
midi_dataframe = MidiDataframe()
for i, file in enumerate(glob.glob("generated_music/xml_gan/*")):
    print(f'Loading file {file}')
    midi_dataframe.load_midi(file, i)
music_list = midi_dataframe.get_slices_note_length(100)
dataset = np.array(music_list)
gan_data = dataset.reshape(dataset.shape[0], dataset.shape[1]*dataset.shape[2])

# WGAN-GP Generated Data
midi_dataframe = MidiDataframe()
for i, file in enumerate(glob.glob("generated_music/xml_wgan-gp/*")):
    print(f'Loading file {file}')
    midi_dataframe.load_midi(file, i)
music_list = midi_dataframe.get_slices_note_length(100)
dataset = np.array(music_list)
wgan_data = dataset.reshape(dataset.shape[0], dataset.shape[1]*dataset.shape[2])

data = np.vstack((real_data, gan_data, wgan_data))
labels = np.hstack((np.zeros(real_data.shape[0]), np.ones(gan_data.shape[0]), np.full(wgan_data.shape[0], 2)))

# Initialize t-SNE
tsne = TSNE(n_components=2, perplexity=5, learning_rate=10)

# Perform t-SNE dimensionality reduction
embeddings = tsne.fit_transform(data)

# Plot the embeddings
plt.scatter(embeddings[labels==0, 0], embeddings[labels==0, 1], c='blue', label='Real', alpha=0.5)
plt.scatter(embeddings[labels==1, 0], embeddings[labels==1, 1], c='orange', label='GAN', alpha=0.5)
plt.scatter(embeddings[labels==2, 0], embeddings[labels==2, 1], c='red', label='WGAN-GP', alpha=0.5)
plt.title("t-SNE")
plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig('plots/t-SNE.png')
plt.cla()
plt.clf()
plt.close()