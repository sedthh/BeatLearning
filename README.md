# Beat Learning
This repository is currently a work in progress. The goal is to have a single generative model that is capable of creating a large variety of beatmaps for any song automatically. Any help is appreciated (especially donation of compute for creating foundation models).

## Current status
Training data generation is done. Model architecture is almsot finished. Currently working on training and evaluating the first models. 

## How does this work?
The model takes chunks of audio data, converts it to tokens (via [Meta's EnCodec](https://github.com/facebookresearch/encodec) model) and uses them to predict 1) a metronome (tempo and offset) and 2) possible beats within that chunk for each track (for example: 1(+1 for holds) track for Taiko, 4(+4 for holds) tracks for Mania). These values are learned via a special loss function. 

The training data is based on [OSU](https://osu.ppy.sh/) [beatmaps](https://osu.ppy.sh/beatmaps/artists) with different modes (OSU, Taiko, Mania, etc.).

## Roadmap
- Convert more beatmaps and train the first models
- Create output data to OSU beatmap converter for automatic stage generation
- Upload models to HuggingFace
- Create proper environment besides requirements.txt
- Create Collab Notebooks for automatic beatmap generation
- Write tests (for data generation too to ensure everything is converted accordingly)
- Write detailed explanation on how the training data is generated, the choice of architecture and how the special loss function works

## Special thanks
The project takes ideas from a previous [AIOSU](https://www.nicksypteras.com/blog/aisu.html) attempt.  
Besides relying on the OSU's wiki, [osu-parser](https://github.com/nojhamster/osu-parser) made the beatmap declarations (especially sliders) more clear. The same goes with [mass downloading](https://github.com/vincentmathis/osu-beatmap-downloade) beatmaps. The model takes cues from [NanoGPT](https://github.com/karpathy/nanoGPT). 
