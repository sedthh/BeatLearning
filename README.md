# Beat Learning
This research is a **work in progress**. Our aim is to build a generative model that can automatically create a wide range of beatmaps for any song. We'd love any help we can get, especially in the form of compute donations for creating foundation models. So, if you're interested, feel free to pitch in! 

*The intention of the AI model is not to diminish the value of manually crafted beatmaps made by individuals. Rather, its goal is twofold: 1) to aid artists in swiftly creating new beatmaps through tool assistance, and 2) to facilitate smaller independent developers in generating content more effortlessly for their rhythm games.*

## How does this work?
The model takes chunks of audio data, converts it to tokens (via [Meta's EnCodec](https://github.com/facebookresearch/encodec) model) and uses them to predict 1) a metronome (tempo and offset) and 2) possible beats within that chunk for each track (for example: 1(+1 for holds) track for Taiko, 4(+4 for holds) tracks for Mania). These values are learned by an encoder-decoder architecture via a special loss function. 

The training data is based on publicly available [OSU](https://osu.ppy.sh/) [beatmaps](https://osu.ppy.sh/beatmaps/artists) with different modes (OSU, Taiko, Mania, etc.).

## Current Status and Roadmap (You Gotta Believe)
The #1 goal is to get the MVP out ASAP (a model that can predict basic beatmaps). 

- OSU data converter ✔️
- Data augmentation for training ⏳ (audio offsets can now be defined, plans for nightcore augmentation)
- Implement LOSSU function for beat learning ✔️
- Transformer model architecture ✔️
- Sanity checks ⏳ (training reconversion to OSU beatmaps passes)
- Mass generate training data ⏳ (have to rerun converter)
- Data loader and (distributed) training scripts ⏳ (basic non-distributed versions in notebooks)
- Train the first models (OSU first, then Mania) ❌
- Model output to OSU beatmap converter ⏳ (very basic version for testing purposes with only OSU mode and hits support)
- Full Pytorch Lightning integration (wrap entire training / validation code) ❌
- Add example notebooks for data generation, training and generating beatmaps ⏳ (data generation is available)
- Create Google Collabs too, at least for beatmap generation from own audio source ❌
- Upload foundation models to Huggingface ❌
- Add simple audio to beatmap converter app to HuggingFace spaces as well ❌
- Create proper environment like a docker image besides just requirements.txt ❌
- Clean up the code, make it more modular so it's easier to adapt to other titles ❌ (beatmap conversion scripts are really messy at the moment)
- Add automatic test cases and proper GIT workflows ❌
- Explain in detail how data conversion and the model works so it's easier to contribute ❌
- Maybe write a paper on it afterwards ❌

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
