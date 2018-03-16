
![Alt text](img/GenderListener.png?raw=true "GenderListener")

# Welcome to the home of TED-plus and GenderListener

**TED-plus** is a dataset of all online TED talks including features from metadata, audio, transcripts, and sound-derived speaker gender labels. Scroll down for the full list of features currently included in TED-plus.

**GenderListener** is a tool that generates speaker gender labels from audio. Use it to add gender labels to any audio data you have.The goal of GenderListener is to make it easier for data scientists and social scientists to explore gender-related trends in speech audio data. GenderListener is based on a Logistic Regression classiffier trained on 1,096 gender-labeld TED talks.

This repository contains all the code used to produce the TED_plus.csv dataset and to build GenderListener.   
TED_plus contains:  
- metadata from all online TED talks
- several derived features to facilitate analysis
- word_per_min information derived from the transcripts
- 34 audio features from the TEDLIUM II audio file set
- gender_name labels from gender_guesser
- gender_sound labels from genderListener

Here are some examples of how to use the resources in this repository (in order from least involved to most involved).

If you want to:  

* Use the enhanced TED dataset
	1. download "data/ted_plus.csv"
		- optionally, check out the exploratory analysis and feature creation in "d0. Enhance_metadata.ipynb"

* Do exploratory analysis on the TED-plus data:
	1. Modify and run **"d0. Enhance_metadata.ipynb"** .   
		-change the input file to  "data/ted_plus.csv"
		- comment out code that creates features which are already included in TED_plus.csv.

* Use PyAudioAnalysis to add audio features to a different dataset
	1.  Extract audio features from your audio files
		- modify and run **"d1. Convert and process audio.ipynb"**
	2. If you have metadata in addition to sound files, merge the metadata and audio features
		- modify and run **"d2. Merge metadata and audio.ipyn"**

* Use genderListener to dd gender_sound labels to a different set of sph or wav files:
	1.  Extract audio features from your audio files
		- modify and run **"d1. Convert and process audio.ipynb"**
	2. Run your audio data through the trained genderListener
		- modify and run notebook **"d4. Run pre-trained genderListener.ipynb"**

* Re-train genderListener on a different set of sph or wav files:
	1.  Extract audio features from your audio files
		- modify and run **"d1. Convert and process audio.ipynb"**
	1. Re-train genderListener with your data's audio features
		- modify and run **"d3. Train and pickle genderListener.ipynb"**
	2. Run your audio data through the trained genderListener
		- modify and run notebook **"d4. Run trained genderListener.ipynb"**


In most cases, the main modification will be to change the input and output filenames.   

Some downloads you may need: 
Raw metadata and transcripts: [Kaggle-Rounak_Banik](https://www.kaggle.com/rounakbanik/ted-talks)  
Raw audio data:[TEDLIUM II](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus)  
pyAudioAnalysis [Python 3 version](https://github.com/ksingla025/pyAudioAnalysis3)  
	- PyAudioAnalysis was created by [Theodoros Giannakopoulos] (http://www.di.uoa.gr/~tyiannak), Postdoc researcher at NCSR Demokritos, Athens, Greece.  
	
	
# TED-plus features:

Original metadata features:
* **name:** The official name of the TED Talk. Includes the title and the speaker.
* **title:** The title of the talk
* **description:** A blurb of what the talk is about.
* **main_speaker:** The first named speaker of the talk.
* **speaker_occupation:** The occupation of the main speaker.
* **num_speaker:** The number of speakers in the talk.
* **duration:** The duration of the talk in seconds.
	- changed to minutes
* **event:** The TED/TEDx event where the talk took place.
* **film_date:** The Unix timestamp of the filming.
	- changed to datetime format
* **published_date:** The Unix timestamp for the publication of the talk on TED.com
	- changed to datetime format
* **comments:** The number of first level comments made on the talk.
* **tags:** The themes associated with the talk.
* **languages:** The number of languages in which the talk is available.
* **ratings:** A stringified dictionary of the various ratings given to the talk (inspiring, fascinating, jaw dropping, etc.)
	- changed to an ordered  dictionary sorted by descending frequency
* **related_talks:** A list of dictionaries of recommended talks to watch next.
* **url:** The URL of the talk.
* **views:** The number of views on the talk.

Enhanced metadata:  
**link:** (html tag to display url as a link that gets opened in a new tab)
**annualTED:** 1 if event is an annual TED event rather than a TEDx event.
**film_year**
**published_year**
**num_speaker_talks**
Topics:   
**technology, 'science','global issues', 'culture', 'design', 'business', 'entertainment','health', 'innovation', 'society'**
Ratings:   
**Fascinating, Courageous, Longwinded, Obnoxious, Jaw-dropping, Inspiring, OK, Beautiful, Funny, Unconvincing, Ingenious, Informative, Confusing, Persuasive**
**speaker_first_name**
**words_per_minute:** Word count from transcript divided by duration in minutes

34 audio features from PyAudioAnalysis:
**'ZCR', 'Energy','EnergyEntropy','SpectralCentroid',  
'SpectralSpread', 'SpectralEntropy','SpectralFlux', 'SpectralRollof',   
'mfcc1','mfcc2','mfcc3','mfcc4',  
'mfccC5','mfcc6','mfcc7','mfcc8',  
'mfcc9','mfcc10','mfcc11','mfcc12', 'mfcc13', 
'Chroma1','Chroma2','Chroma3','Chroma4',  
'Chroma5','Chroma6','Chroma7','Chroma8', 
'Chroma9','Chroma10','Chroma11','Chroma12',  
'Chroma_std'

Name-derived gender labels from gender_guesser:
	- **gender_name:** "male," "mostly-male," "female," "mostly-female," "andy" (for androgenous names), "unknown."
	- **gender_name_class:** 0 for "male" and 1 for "female"

Sound-derived gender labels from gender_listener:
	- **gender_sound:** "male," "female." More categories to be added soon!
