# CS5242-project Instructions

Your task is to build a classification model over a medical image dataset.

All images are resized to 512 * 512 pixels. You are challenged to train a deep learning model for recognizing the category of the test data. You are allowed to upload 3 submissions in maximum per day.

Ranking: 
Private Leader Board: 0.965
Public Leader Board: 0.986


Note: Data for the project not included. 

## Requirements
### python version
- python3

### python packages
- h5py
- pandas
- numpy
- keras
- tensorflow
- scikit-learn
- Pillow
```
pip install - r requirements.txt
```
## Run the program
```
python run.py C:\Users\Delie\Desktop\nus-cs5242\train_data C:\Users\Delie\Desktop\nus-cs5242\test_data
```
or 
```
python run.py /Users/delie.an/Desktop/nus-cs5242/train_data /Users/delie.an/Desktop/nus-cs5242/test_data
```
## Directories and files after running the program
```bash
|   README.md
|   requirements.txt
|   run.py
|   test_result.csv
|
+---ckp
|   |   InceptionResNetV2_512_256_batch_16_weights.hdf5
|   |   MobileNet_512_256_batch_16_999_1_weights.hdf5
|   |   MobileNet_512_256_batch_16_weights.hdf5
|   |   MobileNet_512_256_batch_4_weights.hdf5
|   |   MobileNet_512_batch_16_weights.hdf5
|   |   MobileNet_512_batch_4_weights.hdf5
|   |   Xception_512_256_batch_4_999_001_weights.hdf5
|   |
|   \---csv
|           InceptionResNetV2_512_256_batch_16_prediction_df.csv
|           MobileNet_512_256_batch_16_999_1_prediction_df.csv
|           MobileNet_512_256_batch_16_prediction_df.csv
|           MobileNet_512_256_batch_4_prediction_df.csv
|           MobileNet_512_batch_16_prediction_df.csv
|           MobileNet_512_batch_4_prediction_df.csv
|           Xception_512_256_batch_4_999_001_prediction_df.csv
|
\---code
        main.py
        __init__.py
```
## Dataset directory
```bash
+---test_data
|   \---test_images
|           .png
|
\---train_data
    |   train_label.csv
    |
    \---train_images
            .png
```
