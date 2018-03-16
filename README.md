
![Alt text](img/GenderListener.png?raw=true "GenderListener")

# Welcome to the home of TED-plus and GenderListener

TED-plus is a dataset of all online TED talks including features from metadata, audio, transcripts, and sound-derived speaker gender labels. Scroll down for the full list of features currently included in TED-plus.

GenderListener is a tool that generates speaker gender labels from audio. Use it to add gender labels to any audio data you have.The goal of GenderListener is to make it easier for data scientists and social scientists to explore gender-related trends in speech audio data.

This repository contains all the code used to produce the TED-plus.csv dataset and to build GenderListener. It contains metadata from all online TED talks, plus several derived features that make analysis much easier. It contains a word_per_min information derived from the transcripts. Finally, it also contains 34 audio features from the TEDLIUM II audio file set, gender_name labels from gender_guesser, gender_sound labels from gender_listener.

Here are some examples of how to use the resources in this repository (in order from least involved to most involved).

If you want to:  

* Use the enhanced TED dataset
	1. download "ted-plus.csv"
		- optionally, check out the exploratory analysis and feature creation in "d0. Enhance_metadata.ipynb"

* Do exploratory analysis on the TED-plus data:
	1. Modify and run "d0. Enhance_metadata.ipynb" .   
		-change the input file to  "data/ted-plus.csv"
		- comment out code that creates features which are already included in TED-plus.csv.

* Use PyAudioAnalysis to add audio features to a different dataset
	1.  Extract audio features from your audio files
		- modify and run "d1. Convert and process audio.ipynb"
	2. If you have metadata in addition to sound files, merge the metadata and audio features
		- modify and run "d2. Merge metadata and audio.ipyn"

* Use genderListener to dd gender_sound labels to a different set of sph or wav files:
	1.  Extract audio features from your audio files
		- modify and run "d1. Convert and process audio.ipynb"
	2. Run your audio data through the trained genderListener
		- modify and run notebook "d4. Run pre-trained genderListener.ipynb"

* Re-train genderListener on a different set of sph or wav files:
	1.  Extract audio features from your audio files
		- modify and run "d1. Convert and process audio.ipynb"
	1. Re-train genderListener with your data's audio features
		- modify and run "d3. Train and pickle genderListener.ipynb"
	2. Run your audio data through the trained genderListener
		- modify and run notebook "d4. Run trained genderListener.ipynb"


In most cases, the main modification will be to change the input and output filenames. 
Raw metadata and transcripts: [Kaggle-Rounak_Banik](https://www.kaggle.com/rounakbanik/ted-talks)
Raw audio data:[TEDLIUM II](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus)
pyAudioAnalysis [Python 3 version](https://github.com/ksingla025/pyAudioAnalysis3)
PyAudioAnalysis was created by [Theodoros Giannakopoulos] (http://www.di.uoa.gr/~tyiannak), Postdoc researcher at NCSR Demokritos, Athens, Greece.
