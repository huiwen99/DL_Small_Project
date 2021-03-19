# DL_Small_Project

Done by: Loh De Rong (1003557), Koh Hui Wen (1003593)

In this project, you will have to design a deep learning model, whose task is to assist with the diagnosis of pneumonia, for COVID and non-COVID cases, by using X-ray images of patients.

Our group implemented 2 binary classifier model. Binary Classifier #1 predicts if the input sample is normal or (pneumonia-)infected. If the sample is predicted as infected, it will be passed into Binary Classifier #2 which then predicts if it is COVID or non-COVID.

Refer to the following Jupyter Notebooks for the printouts:
1. `Exploration of Dataset.ipynb`: contains the results about the dataset distribution, data preprocessing and augmentation.
2. `Experiments.ipynb`: contains the experimental results of data augmentation as well as hyperparameter tuning.
3. `Train and Evaluate Model.ipynb`: contains the final training and evaluation results.

Refer to `/model_paths` folder for the saved weights:
1. `bc1.pt`: model path for Binary Classifier #1
2. `bc2.pt`: model path for Binary Classifier #2