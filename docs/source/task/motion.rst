Motion-Related Tasks
==========================

.. _t2m:

Text to Motion Generation
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
OFASys is compatible with training objectives other than autoregressive language modeling.
We now describe how OFASys implements denoising diffusion probabilistic modeling (DDPM), for the task of text-to-motion synthesis.


Here we assume the dataset is a table, where each sample record contains two fields.
Field "mocap" of modality "MOTION" is a BVH file containing motion capture data,
while field "title" of modality "TEXT" is a text sentence describing the captured motion, e.g., "a person walks four steps backward".
Similarly, we can replace "title" with other modalities such as "[AUDIO:...]", "[IMAGE:...]", "[VIDEO:...]" to implement various kinds of conditional synthesis tasks.
And we can simply replace the text with an empty string to implement the task of unconditional motion synthesis, aka., motion prediction.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[TEXT:title] -> [MOTION:mocap,preprocess=motion_6d,adaptor=motion_6d]
