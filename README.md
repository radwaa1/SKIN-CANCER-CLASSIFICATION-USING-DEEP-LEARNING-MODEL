
in skin cancer model:
We used the CNN pre-trained EfficientNetB6 model 
to train on HAM1000 dataset contains 10015 image
after augmentation become 34859 image.
Our initial accuracy was around 87%. And after many experiments as ( resizing, hyper-parameters, augmentations, Adam optimizer with LR=0,0001, and regularizations like: EarlyStopping and BatchNormalization ).
The accuracy increased to 91%, 93%, and finally 95%.
in chat bot model:
Our initial accuracy was 65%, and after many experiments(Augmentation,Adam optimizer with LR=0,001, and regularizations like: EarlyStopping).
The accuracy increased to 91% then 95.66% and 
finally 97.78%.
![image](https://github.com/user-attachments/assets/71a613bd-e484-4960-8e71-1ec09b65f4d3)
