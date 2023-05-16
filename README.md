# More Efficient Training Using Equivariant Neural Networks
### Master thesis at Uppsala University
Convolutional Neural Networks are equivariant to translations; however not defined for other symmetries, and the class output may vary depending on the input's orientation. To mitigate this, the training data is augmented at the cost of increased redundancy in the model. Instead, an Group equivariant convolutional neural network can be built, thus increasing the equivariance to a larger symmetry group. 

In this study, two convolutional neural networks and their respective equivariant counterparts are constructed and applied to the symmetry groups $D_4$ and $C_8$ to explore the impact on performance, batch normalisation, and data augmentation. Results suggest that data augmentation is irrelevant to an equivariant model, and equivariance to more symmetries can slightly improve accuracy. The convolutional neural networks rely heavily on batch normalisation, whereas the equivariant versions can maintain decreased accuracy.

## Brief guidance of the scripts
dataset.py: The dataset handling is performed, via 'mode' a dataset partition can be selected. Remember to download the dataset. \
models.py: All models in script is displayed here. \
main.py: The models are run here, select what models, hyperparameters and dataset you want to load in. Paths (ABS_PATH) will have to be altered to correctly save the results 
