Text-Only Tasks
===================
.. _nlu:

GLUE-Style Natural Language Understanding
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^

GLUE is a benchmark for text understanding, which casts multiple datasets into a unified sentence classification form.
The tasks/datasets include the Corpus of Linguistic Acceptability (CoLA), the Stanford Sentiment Treebank (SST-2),
Microsoft Research Paraphrase Corpus (MRPC), Semantic Textual Similarity Benchmark (STS-B), Quora Question Pairs (QQP),
MultiNLI (MNLI), Question NLI (QNLI), Recognizing Textual Entailment (RTE), and Winograd NLI (WNLI).
A majority of the original datasets are cast as natural lanugage inference (NLI) tasks (identifying entailment, neutral,
contrast relationships) or binary classification tasks (yes/no).
The evaluation metric is Matthew’s Correlation Coefficient for CoLA,
Pearson’s and Spearman’s Correlation Coefficient for STS-B, and accuracy for the rest.
The overall score for this task in evaluation is commonly the arithmetic average over the 8 tasks without WNLI.

Default Template
^^^^^^^^^^^^^^^^

.. code-block:: console

    <BOS> is the sentiment of text " [TEXT:sentence] " positive or negative? <EOS> -> <BOS> is the sentiment of text " [TEXT:sentence,no_loss] " positive or negative? [TEXT:label,closed_set]


Usage
^^^^^^^^^^^^^^^^

.. code-block::

    >>> template = '<BOS> is the sentiment of text " [TEXT:sentence] " positive or negative? <EOS> -> <BOS> is the sentiment of text " [TEXT:sentence,no_loss] " positive or negative? [TEXT:label,closed_set]'
    >>> data = {'sentence': "it 's a charming and often affecting journey"}
    >>> output = model.inference(template, data=data)
    >>> print(output.text)

CASE
^^^^^^^^^^^^^^^^

input:

.. code-block:: console

    it 's a charming and often affecting journey

output:

.. code-block:: console

    positive

.. _summary:

Text Summarization
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^

Text summarization is a natural language generation task, where the model should produce a piece of concise text that covers the main points of the given text.
OFASys currently evaluates text summarization on Gigaword, following related work.
Gigaword for summarization is a naturally-annotated dataset consisting of news articles,
where the first sentence of the article is regarded as the summary for the rest of the first paragraph.
The evaluation metric is ROUGE-L.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

    what is the summary of article " [TEXT:src] "? -> [TEXT:tgt,noise_ratio=0.2]

Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = '<BOS> what is the summary of article " [TEXT:src] "? <EOS> -> <BOS> [TEXT:tgt] '
    >>> data = {'src': "poland 's main opposition party tuesday endorsed president lech walesa in an upcoming "
    ...        "presidential run-off election after a reformed communist won the first round of voting ."}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)

CASE
^^^^^^^^^^^^^^^^^^

input:

.. code-block:: console

    poland 's main opposition party tuesday endorsed president lech walesa in an upcoming
    presidential run-off election after a reformed communist won the first round of voting .

output:

.. code-block:: console

   polish opposition endorses walesa in presidential run-off


.. _niv2:

Natural-Instructions v2
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Natural-instruction v2 is a benchmark of 1,600+ diverse language tasks which evaluates generalization across language tasks by leveraging their language instructions.
It covers 70+ distinct task types, such as tagging, in-filling and rewriting.
These tasks are collected with contributions of NLP practitioners in the community and through an iterative peer review process to ensure their quality.
Natural-Instructions v2 consists of a variety of language tasks and instructions that describe them in plain language.
Each sample contains four fields. Instruction defines a given task in plain language.
This involves a complete definition of how an input text (e.g., a sentence or a document) is expected to be mapped to an output text.
Examples are samples of inputs and correct or wrong outputs to them, along with a short explanation for each.
On average, each sample contains 2.8 positive and 2.4 negative examples.
Src and tgt are a large collection of input-output pairs for each task.
Since this benchmark contains a large collection of tasks, we split the tasks into two subsets: one subset for evaluation and the remaining ones which can be used for supervision.
For evaluation tasks, specifically, we fix a manually-selected collection of 12 categories that represent 154 tasks.
We report ROUGE-L for reporting aggregated performance results across a variety of tasks which is a soft string overlap metric that can be applied to a wide range of text generation tasks.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

    [TEXT:instruction] [TEXT:examples] [TEXT:src] -> [TEXT:tgt,max_length=128]




.. _infill:

Text Infilling
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

    what is the complete text of " [TEXT:text,mask_ratio=0.3] "? -> <BOS> [TEXT:text] <EOS>

