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

## Instructions to run python files
1. `train.py`: Trains either Binary Classifier #1 or Binary Classifier #2  

    To train Binary Classifier #1, run the following command:  
    `%run train.py binary_classifier_1`  
    To train Binary Classifier #2, run the following command:  
    `%run train.py binary_classifier_2`  
    
    Optional parameters:  
    `--epochs`: set number of training epochs  
    `--batch`: set batch size  
    `--lr`: set learning rate  
    `--beta1`: set first momentum term for Adam optimizer  
    `--beta2`: set second momentum term for Adam optimizer  
    `--weight_decay`: set weight decay for regularization on loss function  
    `--gamma`: set gamma for learning rate scheduler  
    `--step_size`: set step size for learning rate scheduler  
    `--cuda`: enable cuda training  
    `--checkpoint`: filepath to a checkpoint to load model  
    `--save_dir`: filepath to save the model
    
2. `evaluate.py`: Evaluates model on validation set  

    After training Binary Classifier #1 and Binary Classifier #2 and saving the models to the `./model_paths` directory, we can evalute the whole model on validation set.   
    For example, Binary Classifier #1 is saved as `"./model_paths/bc1.pt"` and Binary Classifier #2 is saved as `"./model_paths/bc2.pt"`.   
    
    Run the following command:  
    `%run evaluate.py two_binary_classifiers --checkpoint "./model_paths/bc1.pt" "./model_paths/bc2.pt"`  
    
3. `predict.py`: Predicts label of an X-ray image using the model

    Run the following command:  
    `%run predict.py "./model_paths/bc1.pt" "./model_paths/bc2.pt" image_file_path`  
    where `image_file_path` is the file path to the X-ray image.
    
