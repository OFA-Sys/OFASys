Image-Related Tasks
===================


.. _caption:

Image Captioning
-------------------------
Task Introduction
^^^^^^^^^^^^^^^^^^^
Image captioning is a standard vision-language task that requires the model to generate an appropriate text for an image.
We evaluate the multi-modal generation capability of OFAsys on the most widely used MS COCO Caption dataset.
Following previous works, We report CIDEr scores on the Karparthy test split}.

Default Template
^^^^^^^^^^^^^^^^

.. code-block:: console

	[IMAGE:img] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:cap] <EOS>


Usage
^^^^^^^^^^^^^^^^^^^^

.. code-block::

    >>> template = '[IMAGE:img] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:cap] <EOS>'
    >>> data = {'img': "https://www.2008php.com/2014_Website_appreciate/2015-06-22/20150622131649.jpg"}
    >>> output = model.inference(template, data=data)
    >>> print(output)

CASE
^^^^^^^^^^^^^^^^^^

input:

.. image:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/book.jpeg

output:

.. code-block:: console

   a hand is holding an open book


.. _t2i:

Text-to-Image Generation
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Text-to-Image generation has become a task that has attracted more and more attention
of researchers as it demonstrates the excellent creation of neural network models.
Similar to Image Infilling task, we use a VQ-GAN model to convert images into discrete codes,
so that the sequence generator can generate a complete image by generating the code sequence autoregressively.
Following previous works, we train our model on the MS COCO train split and evaluate our model o
n the test split by randomly sampling 30000 images.
As for evaluation, we use CLIP Similarity Score (CLIPSIM) to evaluate the semantic similarity between the query
text and the generated images.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		what is the complete image? caption: [TEXT:text]"? -> [IMAGE,preprocessor=image_vqgan,adaptor=image_vqgan]


Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = 'what is the complete image? caption: [TEXT:text]"? -> [IMAGE,preprocessor=image_vqgan,adaptor=image_vqgan]'
    >>> data = {'text': "a city with tall buildings and a large green park."}
    >>> output = model.inference(instruction, data=data)
    >>> output[0].save_image('0.png')


CASE
^^^^^^^^^^^^^^^^^^

input:

::

	a city with tall buildings and a large green park.

output:

.. image:: https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/image-gen_example.png



.. _vqa:

Visual Question Answering
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Visual question answering (VQA) requires the model to answer questions based on the information of the given image.
We finetune our pretrained model on the dataset VQA-v2.
We evaluate the performance by calculating accuracy.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[IMAGE:image] <BOS> [TEXT:question] <EOS> -> <BOS> [TEXT:answer,closed_set] <EOS>


.. _snlive:

SNLI-VE Visual Entailment
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Visual entailment (VE) is similar to textual entailment. It changes the premise from the text to the image, and judges whether the images matches the sentence. SNLI-VE is a data set of VE tasks which gives images, image captions and premises, and requires the model to judge the relationship between images and premises, and gives one of three outcomes: entailment, neutral, and contradiction.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

	[IMAGE:img] can image and text1 " [TEXT:cap] " imply text2 " [TEXT:hyp] "? ->
	can image and text1 " [TEXT:cap,no_loss] " imply text2 " [TEXT:hyp,no_loss] "? [TEXT:label,closed_set]


.. _imageclass:

Image Classification
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Image classification task requires the model to predict the correct category for the input image.
We evaluate our model on the ILSVRC-2012 ImageNet dataset.
The dataset contains 1K image categories and around 1.3M images.
Each image is manually annotated with one category label among the 1K candidates.
Following previous works, we report the top-1 accuracy on the test set of 50K images.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

	[IMAGE:image] what does the image describe? -> [TEXT:label_name,closed_set]


.. _imginfill:

Image Infilling
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Image infilling task has been proved to be an effective task for both image and multi-model pretraining.
We mask the middle part of the raw images as input, and expect the model learn to restore the masked part
from the corrupted input by generating the discrete codes produced by VQ-GAN models.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

	what is the complete image of "[IMAGE:img,mask_ratio=0.5]"? -> [IMAGE,preprocessor=image_vqgan,adaptor=image_vqgan]

