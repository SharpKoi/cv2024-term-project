# CV2024 Term Project
## Introduction
This repository was created for our term project, including the source code for the convertion from the handwritten mathematical expressions to latex expressions.

## Usage
1. Install all the dependencies
    ```bash
    pip install -r requirements.txt
    ```
2. Download the cleaned dataset from [[Kaggle] Cleaned Aida OCR Dataset](https://kaggle.com/datasets/602f0d4ba1d20809c7e3842121ac6cc0668e73611d6b3713b9a319d1e94f06a1), decompress it, and make sure the directory `data/cleaned_aida/` includes the 10,000 samples, a train/test split and a vocab file.
3. Set environment variable `MLFLOW_TRACKING_URI` to track the metrics during training. If it is not set, it will in default create a mlflow server locally and you can access it from the url: `http://0.0.0.0:5000`.
4. Config the variables in `train.py`
5. Run `train.py`.
    ```bash
    python train.py
    ```

## File Descriptions
- `aida_clean.py`: the script to clean and process the raw aida dataset. Since the raw dataset is too large, we have a cleaned dataset for direct access: [[Kaggle] Cleaned Aida OCR Dataset](https://kaggle.com/datasets/602f0d4ba1d20809c7e3842121ac6cc0668e73611d6b3713b9a319d1e94f06a1), including all the images, masked images, cleaned metadata, train/test splits and the vocabularies.
- `data.py`: includes the definition of `AidaDataset`, which is a pytorch dataset.
- `tokenizers.py`: includes the latex tokenizer, which is used to encode and decode the latex expressions and the token sequences.
- `transforms.py`: includes the transforms such as `FixedAspectResize`, `RandomSpots` and `Binarization`.
  - `FixedAspectResize`: pad and resize the input image to a specific size while fixing the aspect ratio.
  - `RandomSpots`: add some random varying spots on the image to enhance the model performance on noisy binarized images.
  - `Binarization`: binarize the given RGB image.
- `models`: includes the OCR model and the corresponding lightning model we implemented in our project.
- `metrics.py`: includes the definition of `CharacterErrorRate`, which is a metric used to measure the model performance.
- `train.py`: train the model.
- `verify_cleaned_aida`: check the completeness of the cleaned aida dataset.
