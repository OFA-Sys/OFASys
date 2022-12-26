Box-Related Tasks
===================



.. _refcoco:

RefCOCO Visual Grounding
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Visual Grounding requires the model to locate an image region according to a text query.
OFASys formulate this task as a sequence-to-sequence generation task.
In detail, given an image and query, OFASys generates the box sequence
(e.g., <x\ :sub:`1`\ ,y\ :sub:`1`\ ,x\ :sub:`2`\ ,y\ :sub:`2`\ >) in an autoregressive manner.
We perform experiments on RefCOCO, RefCOCO+, and RefCOCOg.
We report the metric Acc@0.5 on the corresponding validation and test sets.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[IMAGE:img] which region does the text "[TEXT:cap]" describe? -> [BOX:patch_boxes]

Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = '[IMAGE:img] which region does the text " [TEXT:cap] " describe? -> [BOX:patch_boxes]'
    >>> data = {'img': "https://www.2008php.com/2014_Website_appreciate/2015-06-22/20150622131649.jpg", 'cap': 'hand'}
    >>> output = model.inference(instruction, data=data)
    >>> output.save_box('0.jpg')


CASE
^^^^^^^^^^^^^^^^^^

input:

.. image:: https://www.2008php.com/2014_Website_appreciate/2015-06-22/20150622131649.jpg

output:

.. image:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/inference_caption_0.jpg


.. _groundedcaption:

Grounded Image Captioning
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Grounded image captioning is an inverse task of visual grounding.
Given an image and a region, the model requires to generate a description about the region.
We use RefCOCO, RefCOCO+, RefCOCOg, and Visual Genome as the pretraining datasets for this task.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[IMAGE:img] what does the region describe? region: [BOX:patch_boxes] -> [TEXT:cap]


.. _od:

Object Detection
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^

Object detection is a common vision task that requires a model to recognize all objects in the image and localize their regions.
We use OpenImages, Object365, Visual Genome, and COCO as the pretraining datasets for this task.

Default Template
^^^^^^^^^^^^^^^^^^
.. code-block:: console

		[IMAGE:img] what are the objects in the image? -> ( [BOX] [TEXT])*

