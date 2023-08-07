import music21 as music
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class MidiDataframe(object):

    def __init__(self):
        self.dataframe = pd.DataFrame(columns=['music_name', 'part_name', 'note', 'duration'])
        self.unique_note_number = 0
        self.unique_duration_number = 0
        self.note_map = {}
        self.duration_map = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.int_scaler = None

    def load_midi(self, file, music_name, remove_accidental=False):
        midi = music.converter.parse(file)
        parts = midi.parts # Separe the midi in parts (Instruments)

        for part_name, part in enumerate(parts): # For each part, get the notes, rests and durations
            for element in part.flat.notesAndRests:
                if (isinstance(element, music.note.Note)):
                    if (remove_accidental):
                        if (element.pitch.accidental == None or str(element.pitch.accidental) == 'natural'): # Dont change natural notes
                            pitch = element.pitch.midi
                        elif (str(element.pitch.accidental) == 'sharp'): # Transform shaps in its natural form
                            pitch = element.pitch.midi - 1
                        else: # Transform flats in its natural form
                            pitch = element.pitch.midi + 1
                    else:
                        pitch = element.pitch.midi
                    row = {'music_name': music_name, 'part_name': part_name, 'note': pitch, 'duration': float(element.duration.quarterLength)}
                elif (isinstance(element, music.note.Rest)):
                    row = {'music_name': music_name, 'part_name': part_name, 'note': -1, 'duration': float(element.duration.quarterLength)}
                elif element.isChord:
                    if (remove_accidental):
                        if (element.root().accidental == None or str(element.root().accidental) == 'natural'): # Dont change natural notes
                            pitch = element.root().midi
                        elif (str(element.root().accidental) == 'sharp'): # Transform shaps in its natural form
                            pitch = element.root().midi - 1
                        else: # Transform flats in its natural form
                            pitch = element.root().midi + 1
                    else:
                        pitch = element.root().midi
                    row = {'music_name': music_name, 'part_name': part_name, 'note': pitch, 'duration': float(element.duration.quarterLength)}
                self.dataframe = pd.concat([self.dataframe, pd.DataFrame(row, index=[0])], ignore_index=True)
            # break # ignore others parts
        
        self.truncate_duration()

    def column_to_map(self, orderby='note'):
        if (orderby == 'note'): # Ordered by note
            self.note_map = sorted(self.dataframe['note'].unique())
            self.duration_map = sorted(self.dataframe['duration'].unique())
        else: # Odered by frequency
            self.note_map = self.dataframe['note'].value_counts().index.tolist()
            self.duration_map = self.dataframe['duration'].value_counts().index.tolist()
        self.note_map = {note: number for number, note in enumerate(self.note_map)}
        self.duration_map = {note: number for number, note in enumerate(self.duration_map)}
        self.dataframe['note'] = self.dataframe['note'].map(self.note_map)
        self.dataframe['duration'] = self.dataframe['duration'].map(self.duration_map)

    def map_to_column(self):
        reverse_note_map = {v: k for k, v in self.note_map.items()}
        reverse_duration_map = {v: k for k, v in self.duration_map.items()}
        self.dataframe['note'] = self.dataframe['note'].abs().astype(int).map(reverse_note_map)
        self.dataframe['duration'] = self.dataframe['duration'].abs().astype(int).map(reverse_duration_map)

    def load_prediction(self, prediction):
        music_generated = []
        for pred in prediction:
            for row in pred:
                music_generated.append({'music_name': '0', 'part_name': '0', 'note': row[0], 'duration': row[1]})
        self.dataframe = pd.DataFrame(music_generated)

    def load_prediction_lstm(self, prediction):
        music_generated = []
        for row in prediction:
            music_generated.append({'music_name': '0', 'part_name': '0', 'note': row[0], 'duration': row[1]})
        self.dataframe = pd.DataFrame(music_generated)

    def scaler_transform(self):
        # self.unique_note_number = len(self.dataframe['note'].unique())
        # self.unique_duration_number = len(self.dataframe['duration'].unique())
        # self.dataframe['note'] = self.dataframe['note'].apply(lambda x: (x - (self.unique_note_number / 2)) / (self.unique_note_number / 2))
        # self.dataframe['duration'] = self.dataframe['duration'].apply(lambda x: (x - (self.unique_duration_number / 2)) / (self.unique_duration_number / 2))
        self.dataframe[['note', 'duration']] = self.scaler.fit_transform(self.dataframe[['note', 'duration']])

    def scaler_inverse_transform(self):
        # self.dataframe['note'] = self.dataframe['note'].apply(lambda x: ((x * self.unique_note_number) + self.unique_note_number))
        # self.dataframe['duration'] = self.dataframe['duration'].apply(lambda x: ((x * self.unique_duration_number) + self.unique_duration_number))
        self.dataframe[['note', 'duration']] = self.scaler.inverse_transform(self.dataframe[['note', 'duration']])

    def int_scaler_transform(self):
        self.dataframe['note'] = self.dataframe['note'].replace(-1, np.nan)
        self.int_scaler = self.dataframe['note'].min(skipna=True)
        self.dataframe.loc[~self.dataframe['note'].isna(), 'note'] -= self.int_scaler
        self.dataframe['note'] = self.dataframe['note'].replace(np.nan, -1)
        self.dataframe['note'] = self.dataframe['note'].astype(int)

    def int_scaler_inverse_transform(self):
        self.dataframe['note'] = self.dataframe['note'].replace(-1, np.nan)
        self.dataframe.loc[~self.dataframe['note'].isna(), 'note'] += self.int_scaler
        self.dataframe['note'] = self.dataframe['note'].replace(np.nan, -1)
        self.dataframe['note'] = self.dataframe['note'].astype(int)

    def get_all_flat(self):
        music_list = []
        for music_name in self.dataframe['music_name'].unique():
            dataframe_music = self.dataframe[self.dataframe['music_name'] == music_name]
            for part_name in dataframe_music['part_name'].unique():
                dataframe_part = self.dataframe[self.dataframe['part_name'] == part_name]
                for row in dataframe_part[['note', 'duration']].values:
                    music_list.append([row[0], row[1]])
                break # Ignore others parts
        return music_list

    def get_slices(self, compass_size):
        size = 4 * compass_size
        actual_size = 0
        slices = []
        for music_name in self.dataframe['music_name'].unique():
            slice_x = pd.DataFrame(columns=list(self.dataframe.columns))
            for _, row in self.dataframe[self.dataframe['music_name']==music_name].iterrows():
                if (row['duration'] + actual_size < size):
                    slice_x = pd.concat([slice_x, pd.DataFrame({'music_name': 'x', 'part_name': 'x', 'note': row['note'], 'duration': row['duration']}, index=[0])])
                    actual_size += row['duration']
                elif (row['duration'] + actual_size == size):
                    slice_x = pd.concat([slice_x, pd.DataFrame({'music_name': 'x', 'part_name': 'x', 'note': row['note'], 'duration': row['duration']}, index=[0])])
                    slices.append(slice_x)
                    actual_size = 0
                    slice_x = pd.DataFrame(columns=list(self.dataframe.columns))
                else:
                    slice_x = pd.concat([slice_x, pd.DataFrame({'music_name': 'x', 'part_name': 'x', 'note': row['note'], 'duration': (size - actual_size)}, index=[0])])
                    slices.append(slice_x)
                    actual_size = round(actual_size + row['duration'] - size, 6)
                    slice_x = pd.DataFrame(columns=list(self.dataframe.columns))
                    slice_x = pd.concat([slice_x, pd.DataFrame({'music_name': 'x', 'part_name': 'x', 'note': row['note'], 'duration': actual_size}, index=[0])])
        return slices

    def get_slices_note_length(self, size):
        slices = []
        for music_name in self.dataframe['music_name'].unique():
            slice_x = []
            i = 0
            for _, row in self.dataframe[self.dataframe['music_name']==music_name].iterrows():
                if i < size:
                    slice_x.append([row['note'], row['duration']])
                    i += 1
                else:
                    slices.append(slice_x)
                    i = 1
                    slice_x = [[row['note'], row['duration']]]
        return slices

    def get_slices_repeated(self, size):
        slices = []
        for music_name in self.dataframe['music_name'].unique():
            dataframe = self.dataframe[self.dataframe['music_name'] == music_name]
            sequences = [(dataframe['note'][i:i+size].tolist(), dataframe['duration'][i:i+size].tolist()) for i in range(len(dataframe) - size + 1)]
            slices += [[list(pair) for pair in zip(seq1, seq2)] for seq1, seq2 in sequences]
        return slices
    
    def find_nearest_truncated(self, value):
        truncated_values = [4, 3, 2, 1.5, 1, 0.5, 0.25, 0.125]
        return min(truncated_values, key=lambda x: abs(x - value))

    def truncate_duration(self):
        self.dataframe['duration'] = self.dataframe['duration'].apply(self.find_nearest_truncated)

    def save_midi(self, method, file_name):
        score = music.stream.Score()
        score.insert(0, music.meter.TimeSignature('4/4'))
        score.insert(0, music.key.Key('C'))
        part = music.stream.Part()
        part.id = 1

        for _, row in self.dataframe[['note', 'duration']].iterrows():
            note_name = row[0]
            duration = row[1]

            if (note_name == -1):
                rest = music.note.Rest()
                rest.duration = music.duration.Duration(duration)
                # rest.channel = 1
                part.append(rest)
            else:
                note = music.note.Note()
                note.pitch = music.pitch.Pitch(note_name)
                note.duration = music.duration.Duration(duration)
                # note.channel = 1
                part.append(note)

        score.insert(0, part)
        score.write('musicxml', f'output/xml_{method}/{file_name}.musicxml')
        