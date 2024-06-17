H-Bridge Fault Detection using CNN
This project uses convolutional neural networks (CNNs) to detect faults in h-bridge circuits. 
The model is trained on a dataset of images of faulty and non-faulty h-bridge circuits. 
The trained model can then be used to predict the class of a new image.

Files:
training.py: Trains the CNN model on the dataset.
inputpredict.py: Uses the trained model to make predictions on new images.
New Dataset: Sample dataset. (100 images per class) ; in practice we used over 1000 images per class.
accuracies.png: table for accuracies of all the optimizers tried. (With real dataset).
ADAMAX-lr=0 001-BS=16.png: Accuracy and Loss vs Epochs graph of best optimizer.
report1.pdf: Detailed description of the project.
saved_hbridge.joblib: Saved model (trained).

Usage:
Run training.py to train the model.
Run inputpredict.py to make predictions on a new image.
