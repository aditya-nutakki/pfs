"""
paper link https://arxiv.org/pdf/1505.04597.pdf

U-Net: Convolutional Networks for Biomedical
Image Segmentation
Olaf Ronneberger, Philipp Fischer, and Thomas Brox
Computer Science Department and BIOSS Centre for Biological Signalling Studies,
University of Freiburg, Germany
ronneber@informatik.uni-freiburg.de,

Abstract. 
There is large consent that successful training of deep net-
works requires many thousand annotated training samples. In this pa-
per, we present a network and training strategy that relies on the strong
use of data augmentation to use the available annotated samples more
efficiently. The architecture consists of a contracting path to capture
context and a symmetric expanding path that enables precise localiza-
tion. We show that such a network can be trained end-to-end from very
few images and outperforms the prior best method (a sliding-window
convolutional network) on the ISBI challenge for segmentation of neu-
ronal structures in electron microscopic stacks. Using the same net-
work trained on transmitted light microscopy images (phase contrast
and DIC) we won the ISBI cell tracking challenge 2015 in these cate-
gories by a large margin. Moreover, the network is fast. Segmentation
of a 512x512 image takes less than a second on a recent GPU. 
"""
