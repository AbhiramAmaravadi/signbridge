# Sign Classifier

This directory contains sign classification model used in this project.

## Model

The model uses two Transformer encoder blocks to classify input hand gestures into 250 ASL signs.

- Preprocessing: temporal interpolation into 64 frames, performed **inside** the TFLite graph (NaN → 0, reshape to `[T, 1629]`, bilinear resize to `[64, 1629]`). Feed the raw sequence; do not resample in Python.
- Input: variable-length landmark sequences `[T, 543, 3]`, float32, in MediaPipe order face (468), left hand (21), pose (33), right hand (21)
- Output: probability distribution over 250 classes `[250]`

## Evaluation

### Split Methond

The labeled dataset was divided into 95% training data and 5% validation data using stratified random split:

``` python
train_df, val_df = train_test_split(
    train,
    test_size=0.05,
    random_state=42,
    stratify=train["label"]
)
```

### Validation Result

- Validation Loss:  1.335006
- Validation Top-1: 67.2523%
- Validation Top-5: 87.5741%

Top-1 stands for the percentage that the correct sign is the prediction.
Top-5 stands for the percentage that the correct sign is among the top-5 predictions.


