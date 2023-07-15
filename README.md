# Beat Learning
This repository is currently a **work in progress**. The goal is to have a single generative model that is capable of creating a large variety of beatmaps for any song automatically. Any help is appreciated (especially donation of compute for creating foundation models).

*The intention of the AI model is not to diminish the value of manually crafted beatmaps made by individuals. Rather, its goal is twofold: 1) to aid artists in swiftly creating new beatmaps through tool assistance, and 2) to facilitate smaller independent developers in generating content more effortlessly for their rhythm games.*

## Current status
Training data generation is done. Model architecture is almsot finished. Currently working on training and evaluating the first models. 

## How does this work?
The model takes chunks of audio data, converts it to tokens (via [Meta's EnCodec](https://github.com/facebookresearch/encodec) model) and uses them to predict 1) a metronome (tempo and offset) and 2) possible beats within that chunk for each track (for example: 1(+1 for holds) track for Taiko, 4(+4 for holds) tracks for Mania). These values are learned via a special loss function. 

The training data is based on publicly available [OSU](https://osu.ppy.sh/) [beatmaps](https://osu.ppy.sh/beatmaps/artists) with different modes (OSU, Taiko, Mania, etc.).

## Roadmap
- Convert more beatmaps and train the first models
- Create output data to OSU beatmap converter for automatic stage generation
- Upload models to HuggingFace
- Create proper environment besides requirements.txt
- Create Collab Notebooks for automatic beatmap generation
- Add simple audio to beatmap converter app to HuggingFace spaces
- Write tests (for data generation too to ensure everything is converted accordingly)
- Write detailed explanation on how the training data is generated, the choice of architecture and how the special loss function works
- Add proper GIT workflows and automatic checks

## Contributor Guidelines
- The goal of the model is to start generating beats right after any audio data is fed into it without any delay. Therefor datasets are clamped in a way that lead in aduio silence and cinematic parts without any hit areas are cut out of training data. 
- Due to the sparsity of training data (too many tokens for too little examples), the models should be as compact and regularized as possible to ensure generalization.
- Any rhythm game related easter-eggs and references are welcome as long as they do not infringe on copyright.
- Generated content should ALWAYS be labelled according to EU regulations. Therefor the metadata of the generated beatmaps must include information about the AI model used.

## Legal
Materials provided here are intended for non-commercial educational use. By accessing the documents here you agree that these materials are for private, educational use only. You accept that you will: 
- NEVER convert or train a model on any copyrighted material! 
- NEVER create beatmaps based on any copyrighted material!

## Special thanks
The project takes ideas from a previous [AIOSU](https://www.nicksypteras.com/blog/aisu.html) attempt.  
Besides relying on the OSU's wiki, [osu-parser](https://github.com/nojhamster/osu-parser) made the beatmap declarations (especially sliders) more clear. The same goes with [mass downloading](https://github.com/vincentmathis/osu-beatmap-downloader) beatmaps. The model takes cues from [NanoGPT](https://github.com/karpathy/nanoGPT). 
