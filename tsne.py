import alphashape
from descartes import PolygonPatch
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
tsne = TSNE(n_components=2, perplexity=500, learning_rate=10)

# Perform t-SNE dimensionality reduction
embeddings = tsne.fit_transform(data)

# Plot the embeddings
fig, ax = plt.subplots(figsize=(10, 8))
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

real_list = [(embeddings[labels==0, 0][i], embeddings[labels==0, 1][i]) for i in range(len(embeddings[labels==0, 0]))]
gan_list = [(embeddings[labels==1, 0][i], embeddings[labels==1, 1][i]) for i in range(len(embeddings[labels==1, 0]))]
wgan_list = [(embeddings[labels==2, 0][i], embeddings[labels==2, 1][i]) for i in range(len(embeddings[labels==2, 0]))]
alpha_shape_real = alphashape.alphashape(real_list, 2.0)
alpha_shape_gan = alphashape.alphashape(gan_list, 2.0)
alpha_shape_wgan = alphashape.alphashape(wgan_list, 2.0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(*zip(*real_list), c='blue', label='Real', alpha=0.5)
ax.scatter(*zip(*gan_list), c='orange', label='GAN', alpha=0.5)
ax.scatter(*zip(*wgan_list), c='red', label='WGAN-GP', alpha=0.5)
ax.add_patch(PolygonPatch(alpha_shape_real, alpha=0.2, color='blue'))
ax.add_patch(PolygonPatch(alpha_shape_gan, alpha=0.2, color='orange'))
ax.add_patch(PolygonPatch(alpha_shape_wgan, alpha=0.2, color='red'))
plt.title("t-SNE")
plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig('plots/t-SNE_convex_hull.png')
plt.cla()
plt.clf()
plt.close()
plt.show()