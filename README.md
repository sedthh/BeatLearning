# Beat Learning
This repository is a **work in progress**. Our aim is to build a cool generative model that can automatically create a wide range of beatmaps for any song. We'd love any help we can get, especially in the form of compute donations for creating foundation models. So, if you're interested, feel free to pitch in! 

*The intention of the AI model is not to diminish the value of manually crafted beatmaps made by individuals. Rather, its goal is twofold: 1) to aid artists in swiftly creating new beatmaps through tool assistance, and 2) to facilitate smaller independent developers in generating content more effortlessly for their rhythm games.*

## Current Status and Roadmap
- OSU data converter ✔️
- Data augmentation for training ⏳ (audio offsets can now be defined, plans for nightcore augmentation)
- Implement LOSSU function for beat learning ✔️
- Transformer model architecture ✔️
- Sanity checks ⏳ (training reconversion to OSU beatmaps passes)
- Mass generate training data ⏳ (have to rerun converter)
- Data loader and training script ⏳ (basic versions in notebooks)
- Training the first model ❌
- Model output to OSU beatmap converter ⏳ (very basic version for testing purposes with only OSU mode and hits support)
- Example notebooks for data generation, training and generating beatmaps ⏳ (data generation is available)
- Upload models to Huggingface ❌
- Add simple audio to beatmap converter app to HuggingFace spaces ❌
- Create proper environment besides requirements.txt ❌
- Clean up the code ❌ (beatmap conversion scripts are really messy at the moment)
- Add automatic test cases and proper GIT workflows ❌

## How does this work?
The model takes chunks of audio data, converts it to tokens (via [Meta's EnCodec](https://github.com/facebookresearch/encodec) model) and uses them to predict 1) a metronome (tempo and offset) and 2) possible beats within that chunk for each track (for example: 1(+1 for holds) track for Taiko, 4(+4 for holds) tracks for Mania). These values are learned via a special loss function. 

The training data is based on publicly available [OSU](https://osu.ppy.sh/) [beatmaps](https://osu.ppy.sh/beatmaps/artists) with different modes (OSU, Taiko, Mania, etc.).

## Guidelines for Contributors
- The main objective of the model is to generate beats immediately after receiving audio data, ensuring no delays. To achieve this, datasets are processed by removing lead-in audio silence and cinematic parts without any hit areas before training.
- Due to the limited amount of training data (too few examples for a large number of tokens), it is crucial to keep the models compact and well-regularized for better generalization.
- We encourage the inclusion of rhythm game-related easter-eggs and references, as long as they do not violate copyright laws.
- All generated content must adhere to EU regulations and should be properly labeled. This means that the metadata of the generated beatmaps must include information about the AI model used.

## Legal
Materials provided here are intended for non-commercial educational use. By accessing the documents here you agree that these materials are for private, educational use only. You accept that you will: 
- NEVER convert or train a model on any copyrighted material! 
- NEVER create beatmaps based on any copyrighted material!

## Special thanks
The project takes ideas from a previous [AIOSU](https://www.nicksypteras.com/blog/aisu.html) attempt.  
Besides relying on the OSU's wiki, [osu-parser](https://github.com/nojhamster/osu-parser) made the beatmap declarations (especially sliders) more clear. The same goes with [mass downloading](https://github.com/vincentmathis/osu-beatmap-downloader) beatmaps. The model takes cues from [NanoGPT](https://github.com/karpathy/nanoGPT) and the OG encoder-decoder architecture. 
