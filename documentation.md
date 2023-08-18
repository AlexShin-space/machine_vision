How to use?

First step (Installation)
install reguirements runnig this command:
    pip3 install -r requirements.txt

Second step (Using)
run the file with path parameter:
    python testing.py "your_test_directory_path_with_images"
After running you will get a list of model predictions in format:
    [character_ASCII_index], [path_to_this_image]


Information
In this project there are two datasets:
    1. Nums from mnist dataset
    2. Letters from file "A_Z Handwritten Data.csv" downloaded from https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
Then I merge this datasets to one in order 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
All comments follow actions in machine_training.py

The model is tested with a test set of digits and the percentage of correct options was 94%
The percentage of correct options in training dataset is 98%.
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_9 (Conv2D)           (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d_9 (MaxPooling  (None, 13, 13, 32)       0         
 2D)                                                             
                                                                 
 dense_9 (Dense)             (None, 13, 13, 256)       8448      
                                                                 
 flatten_6 (Flatten)         (None, 43264)             0         
                                                                 
 dropout_6 (Dropout)         (None, 43264)             0         
                                                                 
 dense_10 (Dense)            (None, 36)                1557540   
                                                                 
=================================================================
Total params: 1,566,308
Trainable params: 1,566,308
Non-trainable params: 0

Machine learning process log
Epoch 1/5
3378/3378 [==============================] - 188s 55ms/step - loss: 0.4850 - accuracy: 0.8629 - val_loss: 0.2241 - val_accuracy: 0.9347
Epoch 2/5
3378/3378 [==============================] - 182s 54ms/step - loss: 0.3047 - accuracy: 0.9121 - val_loss: 0.1938 - val_accuracy: 0.9412
Epoch 3/5
3378/3378 [==============================] - 189s 56ms/step - loss: 0.2603 - accuracy: 0.9246 - val_loss: 0.1703 - val_accuracy: 0.9479
Epoch 4/5
3378/3378 [==============================] - 181s 54ms/step - loss: 0.2375 - accuracy: 0.9306 - val_loss: 0.1593 - val_accuracy: 0.9514
Epoch 5/5
3378/3378 [==============================] - 179s 53ms/step - loss: 0.2248 - accuracy: 0.9334 - val_loss: 0.1639 - val_accuracy: 0.9491

Author:
Oleksandr Shyndin