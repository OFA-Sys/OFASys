=====================================
Define a Task
=====================================

A **Task** in OFASys describes an execution logic specifying which parts of the model should be involved in dealing with certain input-output mapping.
It contains a declarative multi-modal :ref:`Instruction<anchor_instruction>` and a logical plan that supplements model implementation details for a task for certain datasets.

Examples
---------------------

Image Captioning
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    [IMAGE:img] what does the image describe? -> [TEXT:cap]


The instruction above tells what an image captioning task looks like by a single line of code. 
[IMAGE:img] in the left sentence specifies an image input, referred to by the 'img' column in the dataset.
The task is to describe the image, as pointed out by the plain text, which also produces a text sequence that can
be referred to as the 'cap' column in the final output, according to the right sentence.

MNLI Task in Glue Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    can text1 [TEXT:sent1] imply text2 [TEXT:sent2]? ->
    can text1 [TEXT:sent1,no_loss] imply text2 [TEXT:sent2,no_loss] ? [TEXT:label,closed_set]

For the MNLI Task from Glue Benchmark, we may find it useful to copy the encoder input to the beginning of the decoder sentence during training and inference.
However, these copied contents, either the plain text or slots marked by ``no_loss`` are not involved in loss computing.
This is also the case for prompt tuning which prepends some text prompts to the decoder.
Besides, we also use the attribute ``closed_set`` to specify the slot output is restricted to a candidate set.
This is commonly seen especially in open-ended classification, question answering, or detection tasks.

Dynamic Instruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    [IMAGE:img] detect the objects in the image. -> [[BOX][TEXT]]*

Plain text task descriptions can also be written by text slot,
such that a bunch of tasks in the same form can be added by one instruction, e.g., Natural Instruction v2.
For  variable-length input or output, we also support the quantifier "*" attached to a slot, meaning that this slot can repeat several times.
This quantifier is particularly useful for tasks like object detection, whose box-label pair number is highly dynamic.

Zero-shot Inference
=================================

Given an OFASys model and an instruction, we can perform, theoretically, inference on an unseen task, let’s
say image captioning, as follows.

.. code:: python

    >>> instruction = '[IMAGE:img] what does the image describe? -> [TEXT:cap]'
    >>> data = {'img': "oss://ofasys/data/coco/2014/val2014/COCO_val2014_000000391895.jpg"}
    >>> output = model.inference(instruction, data=data)


Register a Task for Training
================================
Everything get a little more complicated during training as there are extra work to associate data with tasks.
To add such a task to OFASys for training, we may need to register it in a python file.
Users can manage related possible configurations and preprocessing of data.
If the task itself has no specific preprocessing, such as the processing of certain fields and the post-processing of inference,
you can directly define a subclass that has no methods and attributes different from the parent class, as follows

.. code:: python

    from ofasys.configure import register_config
    from ofasys.task.base import OFATask, TaskConfig

    @register_config("ofasys.task", "image_classify", dataclass=TaskConfig)
    class ImageClassifyTask(OFATask):
        pass

Start Training
==============

For more details about, see :ref:`Train the Model<anchor_train_the_model>`.
