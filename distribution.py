import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from midi_dataframe import MidiDataframe

# Real Data
real_data = MidiDataframe()
for i, file in enumerate(glob.glob("musicxml/*")):
    print(f'Loading file {file}')
    real_data.load_midi(file, i)
real_data.dataframe = real_data.dataframe.drop(real_data.dataframe[real_data.dataframe['note'] == -1].index)

# GAN Generated Data
gan_data = MidiDataframe()
for i, file in enumerate(glob.glob("generated_music/xml_gan/*")):
    print(f'Loading file {file}')
    gan_data.load_midi(file, i)
gan_data.dataframe = gan_data.dataframe.drop(gan_data.dataframe[gan_data.dataframe['note'] == -1].index)

# WGAN-GP Generated Data
wgan_data = MidiDataframe()
for i, file in enumerate(glob.glob("generated_music/xml_wgan-gp/*")):
    print(f'Loading file {file}')
    wgan_data.load_midi(file, i)
wgan_data.dataframe = wgan_data.dataframe.drop(wgan_data.dataframe[wgan_data.dataframe['note'] == -1].index)


# Notes Frequency
frequency_real = real_data.dataframe['note'].value_counts().sort_index()
proportion_real = frequency_real / frequency_real.sum()

frequency_gan = gan_data.dataframe['note'].value_counts().sort_index()
frequency_gan = frequency_gan[frequency_gan.index.isin(frequency_real.index)]
missing_indexes = frequency_real.index.difference(frequency_gan.index)
frequency_gan = frequency_gan.reindex(frequency_gan.index.union(missing_indexes), fill_value=0)
proportion_gan = frequency_gan / frequency_gan.sum()

frequency_wgan = wgan_data.dataframe['note'].value_counts().sort_index()
frequency_wgan = frequency_wgan[frequency_wgan.index.isin(frequency_real.index)]
missing_indexes = frequency_real.index.difference(frequency_wgan.index)
frequency_wgan = frequency_wgan.reindex(frequency_wgan.index.union(missing_indexes), fill_value=0)
proportion_wgan = frequency_wgan / frequency_wgan.sum()

proportion = pd.DataFrame()
proportion['Real'] = proportion_real
proportion['GAN'] = proportion_gan
proportion['WGAN-GP'] = proportion_wgan
colors = ['blue', 'orange', 'red']
ax = proportion.plot(kind='bar', color=colors, figsize=(20, 10))
plt.xlabel('Notes')
plt.ylabel('Frequency')
plt.title('Notes Frequency')
plt.savefig('plots/notes_frequency.png')
plt.cla()
plt.close()

proportion = pd.DataFrame()
proportion['Real'] = proportion_real
proportion['GAN'] = proportion_gan
proportion['WGAN-GP'] = proportion_wgan
colors = ['blue', 'orange', 'red']
ax = proportion.plot(kind='line', color=colors, figsize=(20, 10))
plt.xlabel('Notes')
plt.ylabel('Frequency')
plt.title('Notes Frequency')
plt.savefig('plots/notes_frequency_line.png')
plt.cla()
plt.close()

print((proportion_real - proportion_gan).abs().sum())
print((proportion_real - proportion_wgan).abs().sum())
print('-----------------------')

# Duration Frequency
frequency_real = real_data.dataframe['duration'].value_counts().sort_index()
proportion_real = frequency_real / frequency_real.sum()

frequency_gan = gan_data.dataframe['duration'].value_counts().sort_index()
frequency_gan = frequency_gan[frequency_gan.index.isin(frequency_real.index)]
missing_indexes = frequency_real.index.difference(frequency_gan.index)
frequency_gan = frequency_gan.reindex(frequency_gan.index.union(missing_indexes), fill_value=0)
proportion_gan = frequency_gan / frequency_gan.sum()

frequency_wgan = wgan_data.dataframe['duration'].value_counts().sort_index()
frequency_wgan = frequency_wgan[frequency_wgan.index.isin(frequency_real.index)]
missing_indexes = frequency_real.index.difference(frequency_wgan.index)
frequency_wgan = frequency_wgan.reindex(frequency_wgan.index.union(missing_indexes), fill_value=0)
proportion_wgan = frequency_wgan / frequency_wgan.sum()

proportion = pd.DataFrame()
proportion['Real'] = proportion_real
proportion['GAN'] = proportion_gan
proportion['WGAN-GP'] = proportion_wgan
colors = ['blue', 'orange', 'red']
ax = proportion.plot(kind='bar', color=colors, figsize=(20, 10))
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Duration Frequency')
plt.savefig('plots/duration_frequency.png')
plt.cla()
plt.close()

proportion = pd.DataFrame()
proportion['Real'] = proportion_real
proportion['GAN'] = proportion_gan
proportion['WGAN-GP'] = proportion_wgan
colors = ['blue', 'orange', 'red']
ax = proportion.plot(kind='line', color=colors, figsize=(20, 10))
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Duration Frequency')
plt.savefig('plots/duration_frequency_line.png')
plt.cla()
plt.close()


print((proportion_real - proportion_gan).abs().sum())
print((proportion_real - proportion_wgan).abs().sum())
print('-----------------------')