Structural Language Tasks
==========================

.. _text2sql:

Text-to-SQL Generation
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Text-to-SQL is a semantic parsing task, aiming to generate executable sql codes according to the command text
and the information of corresponding database information.
In this task, model is supposed not only to truly understand the command text and database,
but also to generate a SQL format code.

In OFASys, we use Exact Matching (EM) metric to evaluate,
measuring whether the generated SQL code are a whole is equivalent to the label SQL query.
Following previous Text-to-SQL works, we first decompose the SQL of both prediction and ground truth as bags of several components (SELECT, WHERE, GROUP BY, ORDER BY, KEYWORDS) and sub-components.
The prediction is correct only if all the components are correct.
The Exact Matching metric are the ratio of correct prediction among all the predictions.

We conduct our experiments on Spider dataset, which contain different complex SQL queries and different complex database in different domains.
It consists of 10,181 questions and 5,693 unique complex SQL queries on 200 databases with multiple tables, covering 138 different domains.
It contain cross-domain cross-database semantic parsing questions.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[TEXT:src] ; structured knowledge: " [TEXT:database] " . generating sql code. -> [TEXT:tgt]


.. _table2text:

Table-to-Text Generation
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Table-to-Text aims to describe a table by natural language. It is a one to many tasks, since there are many ways to describe a table.

The evaluation of Table-to-Text is BLEU, we measure the BLEU score of prediction with the references.

We conduct experiments on DART, which is a triplet component table dataset. We consider it as a three columns, multi rows table without column names. DART has 62659 training set, 5980 valid set and 12552 testing set.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		structured knowledge: " [TEXT:database] "  . how to describe the tripleset ? -> [TEXT:tgt]


.. note::

    we consider the task as a sequence to sequence language task. where database is the table information following the format as,


    .. code-block:: console

           [row1 col1] : [row1 col2] : [row1 col3] | [row2 col1] : [row2 col2] : [row2 col3] | ...


Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = 'structured knowledge: " [TEXT:database] "  . how to describe the tripleset ? -> [TEXT:tgt] '
    >>> data = {'database': database}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)

CASE
^^^^^^

input:

.. code-block:: console

    " 'Atlanta : OFFICIAL_POPULATION : 5,457,831 |
    [TABLECONTEXT] : METROPOLITAN_AREA : Atlanta |
    '5,457,831 : YEAR : 2012 |
    [TABLECONTEXT] : [TITLE] : List of metropolitan areas by population |
    'Atlanta : COUNTRY : United States' "

output:

.. code-block:: console

   atlanta, united states has a population of 5,457,831 in 2012.



.. _tableqa:

Table Question Answering
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^

TableQA is a question answer task according to a given table.
The evaluation of Table2Text is BLEU, we measure the BLEU score of prediction with the references.
We use FeTaQA dataset to evaluate our methods.
FeTaQA is dataset based on 10K Wikipedia pairs (table, question, free-form answer, supporting table cells).
We only use the table, question and free-form answers.


Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

    structured knowledge: " [TEXT:database] "  . what is the answer of the question " [TEXT:src] " ? ->  [TEXT:tgt]

.. note::

    where the "src" is the question, the "database"  is the table and "tgt" is the predict answer.

The table format is the same as Table2Text.


.. _sudoku:

Sudoku
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^

Sudoku task is a common math puzzle game, which filling the blank of a 9*9 tables by digits 1-9 to let every such that
every digit appears exactly once in each row, column and 3*3 box.
Each sudoku has a single unique solution.
We use Solved Accuracy as the evaluation metrics, which means the prediction exactly meeting the requirements is correct.
We use the Sudoku dataset in Kaggle, which contains 10M puzzles among easy to hard.
The dataset is randomly split 1000 samples for validation and 1000 for testing, others are used for training.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		" [TEXT:src] "  .  solve the sudoku .  -> [TEXT:tgt]

.. note::

    the "src" is the sudoku puzzles like (":" split columns and "|" split rows),

        .. code-block:: console

            0 : 8 : 5 : 2 : 3 : 0 : 0 : 7 : 0 | 1 : 4 : 0 : 8 : 0 : 9 : 0 : 0 : 0 | 0 : 7 : 0 : 0 : 1 : 0 : 0 : 0 : 8 | 7 : 0 : 9 : 0 : 0 : 5 : 0 : 0 : 3 | 0 : 0 : 0 : 1 : 6 : 0 : 0 : 0 : 0 | 5 : 0 : 2 : 3 : 0 : 0 : 0 : 1 : 0 | 0 : 0 : 1 : 7 : 4 : 8 : 0 : 5 : 9 | 6 : 5 : 0 : 9 : 0 : 3 : 0 : 0 : 0 | 8 : 9 : 0 : 6 : 0 : 0 : 7 : 0 : 2

    where 0 means blank,  ":" split each digit and "|" split each line.
    The "tgt" is the same format as "src",  replacing the 0 with answers.

