
#%%

import os
import os.path
from pathlib import Path
import src.ads_davidjames9610.ads as ads
import src.ads_davidjames9610.useful as useful
import numpy as np

#%%

datasetFolder = r"/Users/david/Documents/data/speechCommands"

speech_ads = ads.AudioDatastore()
speech_ads.populate(datasetFolder,include_sub_folders=True, label_source=True)

#%%

unique_words = np.unique(speech_ads.labels)

dict_of_words = {}
arr_of_speakers = []
dict_of_speakers = {}

for word in unique_words:
    ads_subset = ads.subset(speech_ads, label=word)
    speakers = []
    for file in ads_subset.files:
        nm = os.path.basename(file)
        nm = nm.split('_')[0]
        speakers.append('a' + nm)
    unique_labels, unique_counts = np.unique(speakers, return_counts=True)
    filtered_labels = (unique_labels[unique_counts > 8])
    if len(filtered_labels) > 1:
        dict_of_words[word] = filtered_labels
        for label in filtered_labels:
            if label in dict_of_speakers:
                dict_of_speakers[label].append(word)
            else:
                dict_of_speakers[label] = [word]

#%%
speakers_dense = []
for key in dict_of_speakers:
    noises = dict_of_speakers[key]
    if len(noises) > 7:
        speakers_dense.append(key)

#%% update ads so that labels are speakers

speakers = []
for file in speech_ads.files:
    nm = os.path.basename(file)
    nm = nm.split('_')[0]
    speakers.append('a' + nm)

speech_ads.set(labels=speakers)

#%%

ads_all = []
for speaker in speakers_dense:
    ads_all.append(ads.subset(speech_ads, label=speaker))

#%%

import librosa
sr = 22050
samples = []
labels = []

for speaker_ads in ads_all:
    
    labels.extend([label for label in speaker_ads.labels]) # all the same
    samples.extend([useful.file_to_audio(file, sr) for file in speaker_ads.files])

# samples = np.concatenate(samples, axis=0)
# labels = np.concatenate(labels, axis=0)

#%%




#%%

# Initialize with the first array
common_elements = arr_of_speakers[0]

# Iterate over the remaining arrays and find common elements
for arr in arr_of_speakers[1:]:
    common_elements = np.intersect1d(common_elements, arr)

#%%

print('')

#%%

ads_subset.set(labels=speakers)

# %%




# %%

# collect speech and labels for 10 speaekers,

