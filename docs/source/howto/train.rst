=======================
Train a Task
=======================


Writing Configurations
========================

OFASys uses a Yaml-based hierarchical configuration system to manage vconfigurations.
Users only need to write several configuration files to start training.
These YAML files can be easily reused or combined to make a multi-task training process.

Task Configuration
---------------------
In the previous chapters we have introduced how to define tasks by simply writing *Instruction* and leave other configuration as default.
Here we will introduce how to write a complete configuration as training requires more information than inference.
Each task configuration file contains information such as preprocess, instruction, evaluation, and criterion.
Suppose we want to implement the image classification task, we need to create a file named image_classify.yaml
in the scripts directory and fill in the following configuration:

.. code:: yaml

    task:
        image_classify:
            instruction:
                template: '[IMAGE:image,preprocess=imagenet] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:label_name,closed_set] <EOS>'

            dataset:
                num_workers: 8
                micro_batch_size: 1
                update_freq: 8

            preprocess:
                imagenet:
                    imagenet_default_mean_and_std: true
                    patch_image_size: 480
                text:
                    ans2label: oss://ofasys/datasets/imagenet-1k/annotations/ans2label.txt

            evaluation:
                metrics:
                    accuracy:
                        target_field: label_name

            criterion:
                label_smoothed_cross_entropy:
                    label_smoothing: 0.1

Among them, the *dataset* field contains parameters such as batch size and update frequency;
the *instruction* field defines the instruction template of the task;
the *preprocess* field declares the preprocessor required for the task,
*ImagenetPreprocessor* and *TextPreprocessor* are required for the image classification task.
the *evaluation* field defines the inference parameters of the model and the evaluation metrics for the task;
the *criterion* field declares that the task is optimized by cross-entropy loss.
The complete available configuration can be found in the dataclass of ``ofasys/task/base.py:TaskConfig``.

Model Configuration
--------------------------------

Model configuration file contains information such as model architecture, model configuration and adaptor configurations.

.. code:: yaml

    model:
        _name: unify
        arch: large
        adaptor:
            image_resnet:
                resnet_type: resnet152
                freeze_resnet: true

        encoder:
            normalize_before: true
            learned_pos: true
        decoder:
            normalize_before: true
            learned_pos: true
        max_source_positions: 1024
        max_target_positions: 1024
        share_decoder_input_output_embed: true
        share_all_embeddings: true
        no_scale_embedding: true
        layernorm_embedding: true
        activation_fn: gelu
        dropout: 0.1
        attention_dropout: 0.0

        encode_drop_path_rate: 0.0
        decode_drop_path_rate: 0.0

        freeze_encoder_embedding: false
        freeze_decoder_embedding: false



Among them, the *_name* field specifies the structure used by the model (OFA, as default);
the *arch* field specifies the model architecture, including embed_dim, num_layers and so on;
the *adaptor* field contains configuration for adaptors, in the image classification task,
we use resnet152 as image adaptor and freeze is during training;
other fields define several common parameters for the model.
The complete available configuration can be found in the dataclass of ``ofasys/model/ofas.py:GeneralistModelConfig``.

.. note::

    All the tasks will share a model configuration YAML file during a multi-task training.

Environment Configuration
---------------------------------
According to the training resources, training can be divided into CPU execution,
single-machine single-GPU training, single-machine multi-GPUs training, and multi-machine multi-GPU training.
The parameters related to the training environment are located in the env_local.yaml environment,
and single-machine and single-GPU training is adopted by default.

* CPU execution

.. code:: yaml

    env:
        runner: local
        nnodes: 1
        nproc_per_node: 1
        cuda_visible_devices: ""

* single-machine single-GPU training

.. code:: yaml

    env:
        runner: local
        nnodes: 1
        nproc_per_node: 1
        cuda_visible_devices: "0"


* single-machine multi-GPU training

.. code:: yaml

    env:
        runner: local
        nnodes: 1
        nproc_per_node: 8
        cuda_visible_devices: "0,1,2,3,4,5,6,7"


* multi-machine multi-GPU training

When using multiple machines and multiple GPUs, the code in the OFASys directory needs to be synchronized to multiple machines,
and the env.rank parameter of each machine needs to be configured separately.

.. code:: yaml

    # env on worker 0
    env:
        runner: local
        nnodes: 1
        nproc_per_node: 8
        cuda_visible_devices: "0,1,2,3,4,5,6,7"
        rank: 0

    # env on worker 1
    env:
        runner: local
        nnodes: 1
        nproc_per_node: 8
        cuda_visible_devices: "0,1,2,3,4,5,6,7"
        rank: 0

Trainer Configuration
---------------------

An example of trainer configuration is given in the following.
The complete available configuration can be found in the dataclass of ``ofasys/configuration/configs.py:TrainerConfig``.

.. code:: yaml

    common:
        fp16: true
        fp16_scale_window: 512
        log_format: simple
        log_interval: 10

    distributed_training:
        find_unused_parameters: true

    optimization:
        max_epoch: 2
        clip_norm: 1.0
        lr: 1e-5
        sentence_avg: false

    optimizer:
        _name: adam
        adam_betas: "(0.9,0.999)"
        adam_eps: 1e-08
        weight_decay: 0.01

    lr_scheduler:
        _name: ofa_polynomial_decay
        warmup_ratio: 0.06

    checkpoint:
        save_interval_updates: 500
        validate_interval_updates: 500

Configuration Inheritance
-------------------------

The configuration can be split into different files to facilitate configuration sharing and independence.
For example, the finetune of the caption task and the finetune of the GLUE task both share many basic configurations (see ``scripts/base.yaml``),
but some task-specific configurations are different (see ``scripts/caption/stage1.yaml`` and ``scripts/glue/cola.yaml``).

OFASys uses the ``_include`` keyword to share the basic configuration. For example in ``scripts/caption/stage1.yaml``:

.. code:: yaml

    # inherit basic configurations
    _include:
        - ../base.yaml
        - ../env_local.yaml

    # override task configuration
    task:
        caption:
            ...

    # override model configuration
    model:
        ...

    # override trainer configuration
    optimization:
        max_epoch: 2
    checkpoint:
        save_dir: oss://ofasys/checkpoints/caption/stage1/${model.arch}/${optimization.max_epoch}_${lr_scheduler.warmup_ratio}
        best_checkpoint_metric: cider

The configuration of caption task includes all baisc configuration in ``scripts/base.yaml`` and environment configuration in ``scripts/env_local.yaml``, and override them by task-specific configuration. For more examples, see ``scripts/glue/*.yaml`` and ``scripts/multitask/stew.yaml``

Training with Distributed Launcher
==================================

Users can use the distributed launcher of OFASys to start training. For example after finishing the yaml configuration, you can launch the training by:

.. code-block:: console

    python ofasys/launch.py scripts/caption/stage1.yaml

Or

.. code-block:: console

    python -m ofasys.launch scripts/caption/stage1.yaml

Users can modify the configuration from the command line after the yaml file.

.. code-block:: console

    python -m ofasys.launch scripts/caption/stage1.yaml --optimization.max_epoch=3 --env.nnodes=2

Furthermore, users can add multiple YAMLs on the command line, with the latter configuration overriding the previous one. For example:

.. code-block:: console

    python -m ofasys.launch \
        scripts/caption/stage1.yaml \
        scripts/snli_ve/train.yaml \
        scripts/multitask/common.yaml \
        scripts/env_local.yaml

The above example is a multi-task training of image captioning (``scripts/caption/stage1.yaml``) and visual entailments (``scripts/snli_ve/train.yaml``).
The multi-task trainer configuration is in ``scripts/multitask/common.yaml``, which shall overrides the trainer configuration of previous task-specific YAMLs.
Users can deploy their environment in ``scripts/env_local.yaml``.

Users can launch all YAMLs in ``scripts/*/*`` in OFASys repo by the launcher to reproduce our experiments.
