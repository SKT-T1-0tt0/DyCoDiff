# Download BEATs Model

The BEATs model file `BEATs_iter3_plus_AS20K.pt` is required when using `--audio_emb_model beats`.

## Download Link

Download from OneDrive:
https://1drv.ms/u/s!AqeByhGUtINrgcpvdNz8-aYim60CIg?e=53V8pg

## Save Location

After downloading, save the file to:
```
saved_ckpts/BEATs_iter3_plus_AS20K.pt
```

## Alternative

If you cannot download BEATs, you can use STFT instead:
```bash
--audio_emb_model STFT
```

This uses STFT spectrogram and doesn't require any additional model files.
