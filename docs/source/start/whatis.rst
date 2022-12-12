========================
OFASys
========================

What is OFASys?
=====================

.. image:: https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/examples/task7.gif

OFASys is a multi-modal multi-task learning system designed to make multi-modal tasks declarative, modular, and task-scalable. With OFASys, it is easy to:

    - Rapidly introduce new multi-modal tasks/datasets by defining a declarative one-line instruction.
    - Develop new or reuse existing modality-specific components.
    - Jointly train multiple multi-modal tasks together without manual processing of multi-modal data collating.

This system aims to allow users to rapidly deploy the model across customized datasets/tasks/modalities,
and provide engineers and researchers with a solution for training one single model/checkpoint to
process multiple (multi-modal) tasks at the near-SOTA level simultaneously.

What does OFASys have?
========================
For now, OFASys supports seven :ref:`modalities<anchor_modalities>`. Including: seven modalities: *TEXT*, *IMAGE*, *VIDEO*, *AUDIO*, *MOTION*, *BOX*, *STRUCT*

OFASys supports more than 20 classes of multi-modal tasks, including:

+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
| Modality  |Task                                         |Dataset       |   Metrics   | OFA+            | OFA+             |  OFA+           |
|           |                                             |              |             | (Specialist)    | (Generalist)     | (Generalist MoE)|
+===========+=============================================+==============+=============+=================+==================+=================+
| Text      |:ref:`NLU<nlu>`                              |GLUE          |  Avg Score ↑|  83.1*          |  \-              |  \-             |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Multiple Tasks<niv2>`                  |Natural       |  ROUGE-L ↑  |  30.49          |  26.97           |  27.74          |
|           |                                             |instruction v2|             |                 |                  |                 |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Text Summarization<summary>`           |Gigaword      |  ROUGE-L ↑  |  34.24          |  34.68           |  33.95          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Text Infilling<infill>`\ :sup:`p`\     |Pile/         |  \-         |  \-             |  \-              |  \-             |
|           |                                             |Wikicorpus/   |             |                 |                  |                 |
|           |                                             |Bookcorpus    |             |                 |                  |                 |
+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
| Image     |:ref:`Image Classification<imageclass>`      |ILSVRC        |  top1 acc ↑ |  83.31          |  72.56           |  78.95          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Visual Entailment<snlive>`             |SnliVE        |  Acc ↑      |  88.88          |  85.84           |  86.18          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Image Captioning<caption>`             |MsCoco        |  Cider ↑    |  134.8          |  122.6           |  125.2          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Visual Question Answering<vqa>`        |VQA-v2        |  VQA score ↑|  78.72          |  68.86           |  72.27          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Text-to-Image Generation<t2i>`         |COCO          |  clip_ti ↑  |  0.317          |  0.289           |  0.294          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Image Infilling<imginfill>` \ :sup:`p`\|  \-          |  \-         |  \-             |  \-              |  \-             |
+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|BOX        |:ref:`Visual Grounding<refcoco>`             |Refcoco       |  Acc @ 0.5 ↑|  88.12          |  80.08           |  83.06          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Grounded Caption<groundedcaption>`     |  \-          |  \-         |  \-             |  \-              |     \-          |
|           |\ :sup:`p`\                                  |              |             |                 |                  |                 |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Object Detection<od>` \ :sup:`p`\      |  \-          |  \-         |  \-             |  \-              |     \-          |
+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
| Video     |:ref:`Video Classification<videoclass>`      |kinetics400   |  Acc ↑      |  74.30          |  64.58           |  69.47          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Video Captioning<videocaption>`        |MSR-VTT       |  Cider ↑    |  70.80          |  59.10           |  63.00          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Video Question Answering<videoqa>`     |MSR-VTT QA    |  VQA score ↑|  42.10          |  41.73           |  40.00          |
+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
| Audio     |:ref:`Automatic Speech Recognition<asr>`     |LibriSpeech   |  WER ↓      |  7.5            |  8.5             |  8.1            |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Text to Speech<tts>`                   | \-           |  mcd loss ↓ |  1.187          |  1.443           |  1.429          |
+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|Structural |:ref:`Text-to-SQL Generation<text2sql>`      |Spider        |Exact Match ↑|  45.70          |  39.20           |  40.50          |
|Language   +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Table-to-Text Generation<table2text>`  |Dart          |  BLEU ↑     |  51.24          |  50.86           |  50.88          |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Table Question Answering<tableqa>`     |Fetaqa        |  BLEU ↑     |  31.56*         |  \-              |    \-           |
|           +---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
|           |:ref:`Sudoku<sudoku>`                        |  \-          | Solved Acc ↑|  99.8*          |  \-              |    \-           |
+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+
| Motion    |:ref:`Text-to-Motion Generation<t2m>`        |AMASS/KIT     |  \-         |  \-             |  \-              |    \-           |
|           |                                             |/AIST++       |             |                 |                  |                 |
+-----------+---------------------------------------------+--------------+-------------+-----------------+------------------+-----------------+

    * Tasks with \ :sup:`p`\  are used in pretraining only.
    * Scores with * are finetuned with a large model.
    * Here, GLUE Benchmark contains seven tasks including COLA, MNLI, MRPC, QNLI, QQP, RTE and SST2.


You can finetune the above individual tasks to achieve some sota results
reported in the `OFA <https://github.com/OFA-Sys/OFA>`_ paper by following :doc:`installation` and :doc:`quickstart`,
or you are free to arbitrarily combine these tasks for larger-scale joint pre-training.
Besides, you can also add new tasks or even new modalities by extending the base classes provided by OFASys.

Contents
==========
The documentation is organized into five sections:

* GET STARTED provides a quick tour of the library and installation instructions to get up and running.

* HOW-TO GUIDES show you how to achieve a specific goal, like how to add a new task or how to write a custom module.

* CONCEPTUAL GUIDES offer more discussion and explanation of the underlying concepts and the design philosophy of OFASys.

* Task Gallery lists all supported tasks.

* API describes all classes and functions.
