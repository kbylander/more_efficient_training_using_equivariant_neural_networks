# Master thesis at Uppsala University
## More Efficient Training Using Equivariant Neural Networks
Convolutional Neural Networks are equivariant to translations; however not defined for other symmetries, and the class output may vary depending on the input's orientation. To mitigate this, the training data is augmented at the cost of increased redundancy in the model. Instead, an Group equivariant convolutional neural network can be built, thus increasing the equivariance to a larger symmetry group. 

In this study, two convolutional neural networks and their respective equivariant counterparts are constructed and applied to the symmetry groups $D_4$ and $C_8$ to explore the impact on performance, batch normalisation, and data augmentation. Results suggest that data augmentation is irrelevant to an equivariant model, and equivariance to more symmetries can slightly improve accuracy. The convolutional neural networks rely heavily on batch normalisation, whereas the equivariant versions can maintain decreased accuracy.

