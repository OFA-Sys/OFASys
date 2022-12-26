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
while field "text" of modality "TEXT" is a text sentence describing the captured motion, e.g., "a person walks four steps backward".
Similarly, we can replace "text" with other modalities such as "[AUDIO:...]", "[IMAGE:...]", "[VIDEO:...]" to implement various kinds of conditional synthesis tasks.
And we can simply replace the text with an empty string to implement the task of unconditional motion synthesis, aka., motion prediction.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

    motion capture: [TEXT:text] -> [MOTION:bvh_frames,preprocess=motion_6d,adaptor=motion_6d]

Usage
^^^^^^^^^^^^^^^^^^^^

Prepare the model, instruction, and text prompts for text-to-motion generation.

.. code:: python

    import torch
    from ofasys import OFASys
    # This checkpoint is for demonstration purposes only, and does not represent the final quality of any project or product.
    # The checkpoint is for research only, and commercial use is prohibited.
    model = OFASys.from_pretrained('http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/model_hub/single_task_motion.pt')
    if torch.cuda.is_available():
        model = model.cuda()
    instruction = 'motion capture: [TEXT:text] -> [MOTION:bvh_frames,preprocess=motion_6d,adaptor=motion_6d]'
    prompts = [
        {'text': 'run then jump'},
        {'text': 'run then jump like dancing'},
    ]

Example 1: Inference without classifier-free guidance. This usage is much simpler and more concise than the classifier-free guided approach described later (see Example 2). However, the generated results tend to correlate poorly with the text prompts. It is thus *NOT* recommended.

.. code:: python

    output = model.inference(instruction, data=prompts)
    output[0].save_as_gif('run_then_jump__no_guide.gif')


Example 2 (*recommended*): Inference with `classifier-free guidance <https://arxiv.org/abs/2207.12598>`_ enabled.
It uses an experimental API for negative prompting and classifier-free guidance.
Classifier-free guidance is implemented by providing the NULL condition (i.e., an empty text) as the negative prompt.

.. code:: python

    guided_prompts = []
    for p in prompts:
        guided_prompts.append(p)
        guided_prompts.append({'text': ''})  # The negative prompt, or an empty string for classifier-free guidance.
    # This API requires the positive and negative prompts be in the same batch, so assert batch_size % 2 == 0.
    output = model.inference(instruction, data=guided_prompts, guidance_weight=3.0, batch_size=2)
    output = output[::2]
    output[0].save_as_gif('run_then_jump__guided.gif')
    output[1].save_as_gif('run_then_jump_like_dancing__guided.gif')
    output[0].save_as_bvh('run_then_jump__guided.bvh')  # Export the result in the BVH format for Blender.

CASE
^^^^^^^^^^^^^^^^^^
The saved result "run_then_jump__guided.gif" should look like below:

.. image:: https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/run_then_jump_guided.gif

The saved result "run_then_jump_like_dancing__guided.gif" should look like below:

.. image:: https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/run_then_jump_like_dancing_guided.gif

The saved result in the `BVH <https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html>`_ file format, "run_then_jump__guided.bvh", can be imported into a 3D animation software such as `Blender <https://www.blender.org/>`_ for rendering:

.. video:: http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/run_then_jump_guided.mp4