# Sign2Speech
Sign2Speech Work In Progress

## Alphabet sound ground-truth [Theo]

For now, we have a very simple dataset of alphabet sounds in WAV that you can download
[here](https://drive.google.com/drive/folders/1Nhr6U8D0oD0q3FAiesV8XSTM94pBeHXE?usp=sharing) as a
tarball or a zip file.

To generate the Mel spectrogram images, set the current folder to the root of the project, download
the archive and execute:
```
# Step 1: extract the WAV archive
tar xzf alphabet.tgz # Or if you're on windows, unzip the ZIP in the current folder
# Step 2: Generate the spectograms
./scripts/gen_alphabet_spectrograms.py data/wav/ data/spec/
# Or something like this on windows, I guess: python scripts/gen_alphabet_spectrograms.py data/wav/ data/spec/
```

Alternatively, you can download [this archive](https://drive.google.com/file/d/122_gEyk1KCZVihON38b7uUtU6INjdAEw/view?usp=sharing), but I don't promise to keep it up to date!

## Training / testing

You only need to download the latest dataset by Frank and place it in `data/train_poses/`.

Everything should be configured to work with the AutoEncoder and the Sign dataset, but you may
override the config via the command line, or add another config (see the 
[hydra documentation](https://hydra.cc/docs/intro)).
To train, simply run the `train.py` script. To test, run the `test.py` script. Simple as that.
