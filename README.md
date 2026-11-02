Introduction:

The most common types of brain tumour are meningiomas,
gliomas, and pituitary tumours [1][3]. Gliomas are the least
likely of the three to be benign, with 75% of malignant tumours being classed as gliomas [2]. Thus, detecting which
type of tumour is present is important for treatment as it can
help to determine the likelihood of the tumour being malignant without any invasive measurements [3]. This is especially important as early detection of cancerous tumours can
greatly increase the patients chance of survival [4].
Many machine learning and AI based models for detecting
these types of tumour have been developed over recent years,
all with high levels of accuracy but varying computational
costs [1][3][5]. This paper will investigate how convolution
neural networks (CNNs) and residual networks (ResNets) perform at identifying these tumour types.



Model Selection:

A CNN architecture was
used to classify the brain scans as this was designed for image
classification [6]. ResNets follow a similar structure but instead use skipped connections to add a correction to the model
at each residual layer. This was also developed for this problem with the aim
of validating that a CNN is a better choice of model for this
problem, as there is not enough data for deep learning to have
a large benefit. Other possible architectures such as multi-layer perceptrons and recurrent neural networks were deemed
unsuitable due to the need for image classification and lack of
sequential data [6]. A mini-batch gradient descent approach
was used rather than stochastic gradient descent to reduce the
computation time of training the networks.

A max pooling aggregate function was used in the CNN
and ResNet. This was chosen over average pooling as it produced higher results across all measured performance metrics.
This makes sense as the difference between tumour types can
be subtle and specific details need to be conserved in order to
construct an accurate model. Adaptive max pooling from the
PyTorch library was also tested, however, the lack of specification of stride and kernel size reduced the accuracy of all of
the models tested.

The sigmoid activation function was initially used as it is
non-linear, and produces values between 0 and 1, which is
appropriate for a classification problem such as this, which
produces probabilities. This stopped the models converging
entirely, and so was replaced with the rectified linear activation function. This did not present the same problem as the
upper limit of the gradients are not capped at 1 and its gradient is always either 0 or 1 which helps to stop the vanishing
gradient problem in back propagation. In order to normalise
the probabilities of belonging to each class for each image, the
softmax function was used on the outputs of each model.

A drop out function was also introduced into both architectures to further reduce over-fitting by forcing the model to
learn more general characteristics of the tumour types so that
when training, the models were not seeing the same images, or
augmented versions of the same images every epoch [9]. The
drop out rate for each neural network was initially set to 0.05
and was increased until the model was no longer over-fitting
when trained with augmented data.



Training The Model:

The data set was split 70 − 15 − 15 between training, validation and test data sets. The same data sets were used for
every model in order to make the performance metrics more
comparable. Also, having the same amount of each tumour
type in the training data set for every model made it easier to
notice biases resulting from uneven class splits in the data.
The cross entropy loss function from PyTorch was used to
calculate the loss of the training and validation data sets after
each epoch. This was chosen as it considers the certainty of
each prediction and penalises uncertain predictions, which is
useful for this classifier problem.

The Adam optimiser was used as it includes learning rate
adjustments for each parameter and performs well for problems with high dimensional parameter spaces, such as this [8].
This was used in conjunction with a learning rate scheduler,
as with Adam alone, each model either plateaued at a relatively high loss (around 0.7) quickly or it would reach lower
validation losses that fluctuated dramatically, rather than producing a low validation loss value and gradually reducing it.
The multiplicative learning rate scheduler from the PyTorch
library was used so the learning rate started high to produce
lower validation losses, and then reduced to gradually reduce
the validation loss value further.

The initial learning rate was chosen as the highest learning rate which didn’t fluctuate to the point of triggering early
stopping within the first 10 epochs so that it would reach as
low validation loss. The function used for the multiplicative
learning rate was chosen by examining the variation of train
and validation loss for many different constant and scheduled
learning rates. Repeated fluctuations of the validation were
seen as indicators of an excessively high learning rate, this
was then used to alter the function so that it reduced to a lower
learning rate at this epoch. This was trialled and refined repeatedly until the validation loss curve followed a near smooth
descent without plateauing.

To reduce the impact of over-fitting, early stopping was
used so the model stops training once the validation loss starts
to plateau. The state of the model when the lowest validation
loss was produced is restored so that the over-fitting that occurs before the network notices the plateau does not affect the
final results.

Data augmentation was introduced to reduce over-fitting so
the networks would see slightly different training images at
every epoch. This forced the networks to focus on the identifying features of the tumours, which were not altered through
augmentation, rather than effectively learning the labels of
each training image. The augmentation transformations were
chosen as translations and minor rotations to leave the images
as realistic tumour scans. The range of the rotations and translations applied was chosen by running the code with a range
of parameters and using those which minimised over-fitting.
The degree of over-fitting was determined by examining the
loss function plots as well as the training and validation data. Both architectures were trained with the
original training data set and with augmented training data to
validate that this decision improved accuracy.

The amount of epochs was chosen as the average epoch
when the validation loss plateaued to limit over-fitting. 
amount of consecutive epochs where the validation loss did
not improve was used to select the patience of each model;
the smallest consecutive amount which did not occur during
any minor plateaus earlier in training was chosen.



Model Performance:

The true positive rate (TPR) and false positive rate (FPR)
of each model was calculated as it considers all three classes
at once. A confusion matrix was produced for each model,
which shows the most common false positives. For a more
detailed analysis, the receiver operating characteristic (ROC)
curve of each model for meningioma, glioma, and pituitary tumours, and the areas under these curves (AUCs), were plotted
and calculated. This showed the TPR and FPR across many
thresholds for each class, providing more detail than the previously calculated TPR and FPR.[6].

It can be seen that the data augmentation
successfully reduced the models’ over-fitting, and, for both
CNN and ResNet architectures, this allowed the validation
loss to reach lower values. Additionally, these plots also show
that the ResNet architecture produced training and validation
losses which fluctuated more than the CNN. This is likely due
to the batch size used in the model causing excess noise in the
batch normalisation process [10].


The ResNet and CNN architectures trained on augmented
data performed similarly, as shown in Table I. The confusion matrices also suggest that the ResNet
models were worse at identifying pituitary tumours than the
CNN models, but more effective at identifying meningioma
and glioma tumours, when trained with augmented data.



Architecture:

CNN ResNet Training data Original Augmented Original Augmented

TPR 0.920 0.905 0.866 0.895

Meningioma AUC 0.977 0.967 0.939 0.963

Glioma AUC 0.985 0.982 0.957 0.978

Pituitary AUC 0.999 0.997 0.989 0.988

TABLE I: The true positive rates and the area under ROC
curve for each type of tumour for each model.

The AUCs displayed in Table I show that every model performed the best when identifying pituitary tumours and the worst at identifying meningioma tumours. As
pituitary tumours are characterised by their location in small
pituitary gland, it makes sense that they would be the easiest to
identify from the training data [11]. The meningioma tumours
likely had the least accuracy due to the smaller amount of images in the dataset when compared with glioma and pituitary images.

The ResNet trained with augmented data had worse TPR
than the CNN, regardless of whether augmented data was
used. This suggests that the CNN correctly predicted the categories of more images in the test data set, however, the ResNet
Contrary to expectations, the CNN without augmented data
performed the best by every performance metric displayed inTable I.



Discussion:

The performance metrics of all four models agree with previous literature for CNN and shallow ResNet architectures,
with TPR around 0.85 - 0.95 and AUCs > 0.9 [1]. The
ResNet produced a lower TPR and AUC for all tumour types,
as expected, and took over twice the training time compared
to the CNN, which was not expected from previous studies
[1]. This is likely due to the lack of depth in this network
as it consists of 15 layers. Additionally the data set used is
likely too small to reap the benefits of ResNet over CNN, and
the vanishing gradient problem is controlled by the rectified
linear activation function, causing CNNs to be as, if not more
effective. Both CNN models performed better than the ResNet
models, which validates the choice of the CNN model. [7]
When trained with the original non-augmented training
data set, the ResNet performed worse on every performance
metric shown in Table I. Also, its validation
loss plateaus higher than all other models. As the ResNet is a
deeper model, it begins over-fitting, and hence the validation
loss begins plateauing, earlier than it does for the CNN.

In contrast, the data augmentation successfully stopped both models over-fitting and led to much higher
TPRs and AUCs for the ResNet but not the CNN. However,
the method of choosing augmentation parameters does not
guarantee that the full tumour is still included in all of the images. Instead, more images from the augmented data set could
be displayed so there is a greater chance of noticing any poor
augmentations, however this would be a very time consuming
process due to the amount of images. A better approach may
be to use a data set which contains the coordinates of the tumour on each image so that the size of possible translations
and rotations could be limited to ensure no tumours would be
altered by the augmentation.

Also the validation loss, when not trained
on augmented data, plateaued long before early stopping was
triggered. This is because small decreases in the validation
loss can be seen occasionally throughout the plateau. This
suggests that early stopping should be improved by introducing a higher minimum difference between loss values to be
considered as decreasing rather than plateauing.

In addition, all four models had the lowest AUC for meningiomas and the highest for pituitary tumours. This could be
a result of the training set having an uneven split of tumour
types, as all the models had the same amount of each tumour
type in the training set. For future models, the augmented data
could be used to give the models an equal amount of data of
each tumour by augmenting the same images multiple times
in different ways.

Differences in the performances of the CNN and ResNet
architectures could also be due to one’s hyperparameters being optimised less effectively than the other. To mitigate this
affecting conclusions, in the future one automated hyperparameter optimisation method, such as Bayesian optimisation,
should be used on all architectures to ensure they are all similarly optimised. [12]

As can be seen from Table I, the CNN model
trained without augmented data displayed over-fitting, yet performed better than the augmented data set. This is likely because the data set contains multiple slices of each patients’
brain scan, and so the test, validation and training data sets
may contain different slices from the same scans which have
similar features. This means that over-fitting on the training
data set translates to high accuracy on the test data set which
would not be produced for a different data set. This could be
improved by retrieving the patient IDs of each slice from the
original data set on MATLab, so the test, validation and training data sets can be made to not contain any data from the
same patients as each other.

The performance metrics displayed in Table I varied each
time the code was ran due to the differences in the test, train
and validation data sets. In order to get more accurate values
k-fold cross-validation could be used, in which the dataset is
split into k groups, and is trained k times, each time leaving a
different one of these groups for validation, and using the rest
for training. This may give results that better represent how
the models would perform on real world data, however the
similarity of images from multiple slices of each scan would
need also need to be rectified to achieve this.[13]

Also, the accuracy could potentially be improved by experimenting more with the hyperparameters of Adam, such
as weight decay. The need for a learning scheduler may also
have been negated by enough optimisation of these hyperparameters, which would make the model simpler. Alternative
optimisers could also be implemented, such as MADGRAD
which consistently outperforms Adam and is not specially designed for any method or architecture, making it more versatile [14].

This model could also be extended to included a ’no tumour’ category trained on healthy brain scans, making the
model more versatile and useful for real world patient diagnosis.
Pretrained versions of ResNet-50 have been shown to be
much more effective, producing AUCs of 1.00 across all categories, while keeping the computational cost reasonably low
[1]. Further research could be extended to include these models to improve accuracy.

Also, attention mechanisms could be included to improve
performance by making the model focus on the most important tumour characteristics for classification [15]. A Vision
Transformer (ViT), which replaces the traditional CNN convolution layers with attention layers to improve accuracy with
far less computational cost [15][16] could also be employed
for this problem, as it has been shown to produce 99% accuracy and AUCs > 0.995 [17].



References:

[1] Lightweight Transfer Learning Models for Multi-Class Brain
Tumor Classification: Glioma, Meningioma, Pituitary Tu-
mors, and No Tumor MRI Screening., A. Gorenshtein, T.
Liba, & A. & Goren, J Digit Imaging. Inform. med. (2025).
https://doi.org/10.1007/s10278-025-01686-1

[2] CBTRUS Statistical Report: Primary Brain and Other Cen-
tral Nervous System Tumors Diagnosed in the United States
in 2014–2018., Q. T. Ostrom, G. Cioffi, K. Waite, C.
Kruchko, J. S. Barnholtz-Sloan, Neuro-Oncology 23, (2021).
https://doi.org/10.1093/neuonc/noab200

[3] Texture, Morphology, and Statistical Analysis to Differentiate
Primary Brain Tumors on Two-Dimensional Magnetic Reso-
nance Imaging Scans Using Artificial Intelligence Techniques,
S. Bhattacharjee, D. Prakash, C. Kim, H. Kim, H. Choi, Korean
Society of Medical Informatics, doi: 10.4258/hir.2022.28.1.46,
(2022).

[4] Screening — Brain and spinal cord tumours — Can-
cer Research UK. (2019). Cancerresearchuk.org.
https://www.cancerresearchuk.org/about-cancer/brain-
tumours/getting-diagnosed/screening

[5] TumorGANet: A Transfer Learning and Generative Adversar-
ial Network- Based Data Augmentation Model for Brain Tumor
Classification, A. Nag, et al, IEEE Access 12, 103060-103081,
(2024). https://doi.org/10.1109/ACCESS.2024.3429633.

[6] Advanced Computational Physics & Machine Learning Course
Notes, J. Brooke, S. Paramesvaran, (2025).

[7] Deep Residual Learning for Image Recognition, Kaiming He,
Xiangyu Zhang, Shaoqing Ren, Jian Sun, arXiv:1512.03385,
(2015).

[8] Adam: A Method for Stochastic Optimization., D.
P. Kingma, & J. Ba, (2014, December 22). ArXiv.
https://arxiv.org/abs/1412.6980

[9] Improving neural networks by preventing co-adaptation of fea-
ture detectors., G. E. Hinton, N. Srivastava, A. Krizhevsky,
I. Sutskever, & R. R. Salakhutdinov, ArXiv:1207.0580,
https://arxiv.org/abs/1207.0580, (2012).

[10] Delving into the Estimation Shift of Batch Normalization in
a Network., L. Huang, Y. Zhou, T. Wang, J. Luo, & X. Liu,
(2022).

[11] Types of brain tumour, www.macmillan.org.uk. Available
at: https://www.macmillan.org.uk/cancer-information-and-
support/brain-tumour/types-of-brain-tumour.

[12] Bayesian Optimization for Hyperparameters Tuning in Neural
Networks. G. Onorato, arXiv:241021886, (2024).

[13] An introduction to statistical learning : with applications in R
(Second edition), G. James, D. Witten, T. Hastie, R. Tibshirani,
(2021). Springer. https://doi.org/10.1007/978-1-0716-1418-1

[14] Adaptivity without Compromise: A Momentumized, Adaptive,
Dual Averaged Gradient Method for Stochastic Optimization.,
A. Defazio, S. Jelassi, arXiv:2102:11075, (2021).

[15] Attention mechanism.IBM, Ibm.com., (2024).
https://www.ibm.com/think/topics/attention-mechanism

[16] An Image is Worth 16x16 Words: Transformers for Im-
age Recognition at Scale. A. Dosovitskiy, L. Beyer,
A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner,
M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J.
Uszkoreit, & N. Houlsby, ArXiv:2010.11929 [Cs] (2020).
https://arxiv.org/abs/2010.11929

[17] Enhancing Brain Tumor Classification Using Vision Trans-
formers with Colormap-Based Feature Representation
on BRISC2025 Dataset., F. Ahmed, ArXiv.org., (2025).
https://arxiv.org/abs/2603.21234
