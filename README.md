# Chest x-ray diagnosis with deep-learning CNN on the CheXpert dataset

Multi-class models are trained for chest x-ray diagnosis of 14 observations using different deep learning architectures and a large dataset of chest x-ray images called CheXpert. We have used the DenseNet-121 architecture
and used transfer learning to train the model, the training was carried out using kaggle kernels.

While a good accuracy is achieved on testset data, the F1 scores on a few observations were low. This was an indication of model robustness issue for a few class predictions. Further analysis of the data indicates an unbalance between available data for those observations with low F1 scores.

Instead of using the complete high-resolution dataset (~430G) we use an up-sampling approach to balance the training data. This results in a significant improvement in both accuracy and F1 scores over the testset data. Finally, a gradient weighted Class Activation Map is applied to localize the highest probability observation for
a given x-ray image input.

[Data](https://stanfordmlgroup.github.io/competitions/chexpert/) :
CheXpert is a large public dataset for chest radiograph interpretation, consisting of 224,316 chest radiographs of 65,240 patients. (Down sampled dataset size: ~11G)

![Classes](/images/classes.png)

Image source: ([link](https://stanfordmlgroup.github.io/competitions/chexpert/))

### Test Score and Accuracy:

- DenseNet_Basic (unbalanced, without up-sampling):
    Score: 0.44067
    Accuracy: 0.80654

- DenseNet_Balanced (up-sampling applied);
    Score: 0.186
    Accuracy: 0.88
