# Sign2Speech
Sign2Speech Work In Progress

## Alphabet sound ground-truth [Theo]

For now, we have a very simple dataset of alphabet sounds in WAV that you can download
[here](https://drive.google.com/drive/folders/1Nhr6U8D0oD0q3FAiesV8XSTM94pBeHXE?usp=sharing) as a
tarball or a zip file.

To generate the Mel spectrogram images, set the current folder to the root of the project, download
the archive and execute:
```
tar xzf alphabet.tgz
# Or if you're on windows you'll figure out how to unzip the archive:
unzip alphabet.zip
cd dataset
./gen_alphabet_spectrograms.py wav/ spec/
# Or something like this on windows, I guess:
python gen_alphabet_spectrograms.py wav/ spec/
```

Or simply download [this archive](https://drive.google.com/file/d/122_gEyk1KCZVihON38b7uUtU6INjdAEw/view?usp=sharing), but I don't promise to keep it up to date!
