# IBL neural encoding and decoding from video latents

This guide describes how to run encoding and decoding using extracted video latents.

To start, install the following packages:
```{bash}
pip install ray facemap==1.0.7 torcheval accelerate
```

## Encoding or Decoding

Run the following command to automatically fit RRR and TCN encoding / decoding models using `Ray Tune`:
```{bash}
python src/test.py \
    --eid $EID \
    --neural_input_dir $PRECACHED_NEURAL_ROOT/$SESSION_ID \
    --latent_input_dir $PRECACHED_LATENT_ROOT/$SESSION_ID \
    --eval_task encoding # or decoding
```
**Note**: 
- Remember to request 1 GPU and 2 CPUs when submitting jobs for `Ray Tune`.
- The encoding results will be automatically saved to `latent_input_dir`.

