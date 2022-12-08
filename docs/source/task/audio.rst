Audio-Related Tasks
=====================

.. _asr:

Automatic Speech Recognition (ASR)
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^

Automatic Speech Recognition~(ASR) is the task of converting speech into sequences of discrete semantic tokens. We evaluate our model on the Librispeech and AISHELL-1 dataset. The Librispeech dataset contains 1000 hours of speech in English sampled at 16 kHz. The AISHELL-1 dataset contains 178 hours of Mandarin speech sampled at 16 kHz.

Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[AUDIO:wav] <BOS> what is the text corresponding to the voice? <EOS> -> [TEXT:text,preprocess=text_phone,add_bos,add_eos]

Usage
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> instruction = '[AUDIO:wav] <BOS> what is the text corresponding to the voice? <EOS> -> [TEXT:text,preprocess=text_phone,add_bos,add_eos]'
    >>> data = {'wav': 'oss://ofasys/data/librispeech/dev-clean/1272/128104/1272-128104-0001.flac'}
    >>> output = model.inference(instruction, data=data)
    >>> print(output.text)


Case
^^^^^^^^^^^^^^^^^^^^


input:

.. raw:: html

    <audio controls="controls">
      <source src="http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/librispeech/dev-clean/1272/128104/1272-128104-0001.flac" type="audio/wav">
      Your browser does not support the <code>audio</code> element.
    </audio>

output:

::

   NOR IS MISTER QUILTERS MANNER LESS INTERESTING THAN HIS MATTER

.. _tts:

Text-to-speech (TTS)
-------------------------------------------

Task Introduction
^^^^^^^^^^^^^^^^^^^
Text-to-speech~(TTS) is the task of generating speech from input text.
We evaluate our model on the LJSpeech and BZNSYP datasets. The LJSpeech dataset contains 24 hours of English audio of a single speaker reading passages with a sample rate of 22050 Hz. The BZNSYP dataset includes 12 hours of Mandarin audio sampled at 48 kHz from a single speaker.


Default Template
^^^^^^^^^^^^^^^^
.. code-block:: console

		[PHONE:text] -> [AUDIO:fbank,adaptor=audio_tgt_fbank]

