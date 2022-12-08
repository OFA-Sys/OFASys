=========================================
Add a Custom Module
=========================================

Users can easily build experiments with their new ideas by extending base classes.
Configurations of the new components can be registered into the system by using the decorator ``@register_config``.
After registration, users can specify and use their newly added modules in the instruction just like the system preset modules.

This section will fully introduce how to add an adapter in OFASys as an example of custom development.

Create a new adaptor
===================================
Here we will replicate the steps to add a ViT adaptor for Image Modality as a tutorial.

Inherit ``BaseAdaptor``
---------------------------
We provide a base class called ``BaseAdaptor`` for both **InputAdaptor** and **OutputAdaptor**.
This class contains three main methods: ``__init__()``, ``forward()`` and ``forward_output()``.

.. code:: python

    class BaseAdaptor(torch.nn.Module):
        def __init__(
            self,
            embed_tokens: Embedding,
            dictionary: Dictionary,
            is_src: bool,
            general_adaptor,
            cfg: BaseAdaptorConfig,
        ):
        super().__init__()

        @abstractmethod
        def forward(self, inputs: Union[Slot, List[Slot]], **kwargs) -> AdaptorOutput:
            """
            The Adaptor work as the InputAdaptor, takes corresponding data in tensor format as input,
            and then output sequences in the same format -ref **AdaptorOutput**.

            Args:
                inputs (Slot): preprocessed input data.

            Returns:
                AdaptorOutput:
                    adaptor_output: adaptor output for the input slot.
            """
            raise NotImplementedError

        def forward_output(self, x: Tensor, extra: Dict[str, Any], slot: Slot, **kwargs):
            """
            The Adaptor work as the OutputAdaptor, takes hidden states from model as input,
            and then output the modality data in their own form, e.g. probs on vocabulary.

            Args:
                x (Tensor): hidden states from model in the shape of
                 ``(batch_size, seq_length, embed_dim)``
                extra (Dict[str, Any]): extra model output information.
                slot (Slot):  input preprocessed data.

            Returns:
                tuple:
                    - x (Tensor): modality data in Tensor form.
                    - extra (Dict[str, Any]): model output with any modality-specific information.
            """
            return x, extra

We start the custom development by extending the base class and then implement these main methods according to actual requirements.

.. code:: python

    class ImageViTAdaptor(BaseAdaptor):

Implement ViT Adaptor
--------------------------
Since we only expect to use ViT as an Input Adapter, we only need to implement two main methods: ``__init__()`` and ``forward()``.
First, we implement the init method. As shown above, the base class takes five parameters to init

    * embed_tokens: Embedding matrix for the global vocabulary.
    * dictionary: Global vocabulary shared by all tasks.
    * is_src: Which part of the model will this adaptor be used.
    * general_adaptor: Instance of GeneralAdaptor.
    * cfg: Configuration of this adaptor.

As ``GeneralAdaptor`` will properly pass in these parameters when initializing these adaptors, we don't pay much
efforts on the parameter passing process here.
We just need to place configuration parameters in the AdaptorConfig class.

.. code:: python

    def __init__(
        self,
        embed_tokens: Embedding,
        dictionary: Dictionary,
        is_src: bool,
        general_adaptor,
        cfg: ImageVitAdaptorConfig,
    ):
        super().__init__(embed_tokens, dictionary, is_src, general_adaptor, cfg)
        vit_backbone = {
            'vit_base': vit_base,
            'vit_large': vit_large,
            'vit_large_336': vit_large_336,
            'vit_huge': vit_huge,
        }[cfg.vit_type]
        self.embed_images = vit_backbone(cfg.vit_drop_path_rate)
        self.image_proj = Linear(self.embed_images.width, cfg.embed_dim)
        if self.cfg.pretrained_ckpt_path:
            local_model_path = cached_path(self.cfg.pretrained_ckpt_path)
            sd = torch.load(local_model_path, map_location="cpu")
            logger.info(
                f'loading adaptor ckpt from {self.cfg.pretrained_ckpt_path} , {self.embed_images.load_state_dict(sd)}'
            )

Then we implement the ``forward()`` method, which takes image slot as input, extract features using ViT backbone,
and finally, return in the standard format ``AdaptorOutput``.

.. code:: python

    def forward(self, slot: Slot, **kwargs) -> AdaptorOutput:
        """
        Args:
            slot (Slot): ModalityType.IMAGE
        Returns:
            AdaptorOutput:
                - **embed** (Tensor): the processed embedding for OFA of
                  shape `(src_len, batch, embed_dim)`
                - **padding_masks** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **pos_embedding** (Tensor): the position embeddings
                  of shape `(batch, src_len, embed_dim)`
                - **self_attn_bias** (Tensor): attention bias in self attention
                 of shape `(layer_num, batch, num_attention_heads, src_len, src_len)`.
        """
        assert slot.modality == ModalityType.IMAGE
        sample_patch_num = kwargs.pop('sample_patch_num', None)
        (image_embed, image_num_patches, image_padding_mask,
         image_position_ids, image_pos_embed) = \
            self.get_patch_images_info(slot.value, sample_patch_num)
        image_embed = self.image_proj(image_embed)

        batch_size, seq_length = image_embed.size()[:2]
        self_attn_bias = []
        if self.cfg.use_self_attn_bias:
            num_rel_pos_tables = 1 if self.cfg.share_attn_bias else self.num_layers
            for idx, layer in enumerate(range(num_rel_pos_tables)):
                values = self.get_rel_pos_bias(batch_size, seq_length, idx,
                                               image_position_ids=image_position_ids)
                self_attn_bias.append(values)

        return AdaptorOutput(
            image_embed, image_padding_mask, image_pos_embed, self_attn_bias)


.. note::
    Only the main code is shown here, and some codes are omitted for simplification.

Add ViT Adaptor Config
------------------------------

As shown in ``__init__()`` and ``forward()``, we need three extra parameters: ``vit_type``, ``vit_drop_path_rate`` and
``pretrained_ckpt_path``. So we also extend the ``BaseAdaptorConfig`` class and create a new ``ImageVitAdaptorConfig``.

.. code:: python

    @dataclass
    class ImageVitAdaptorConfig(BaseAdaptorConfig):
        vit_type: ChoiceEnum(['vit_base', 'vit_large', 'vit_large_336', 'vit_huge']) = field(
            default='vit_base', metadata={"help": "vit type"},
        )
        vit_drop_path_rate: float = field(
            default=0., metadata={"help": "resnet drop path rate"},
        )
        pretrained_ckpt_path: str = field(
            default="", metadata={"help": "path of pretrained ckpt"}
        )

Register the Adaptor and Config
-------------------------------
We can register the newly added Adaptor and Config classes by the decorator ``@register_config``.

.. code:: python

    @register_config("ofasys.adaptor", "image_vit", ImageViTAdaptorConfig)
    class ImageViTAdaptor(BaseAdaptor):

Use it in an Instruction
---------------------------

Now that all the development work is done, it's time to use it in a task!
Still take the caption task we mentioned before as an example.
The original content of ``caption.yaml`` is:

.. code:: yaml

    task_name: caption
    instruction:
         - '[IMAGE:image] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:caption] <EOS>'

We can change to use ViT as the image adaptor instead of ResNet, by simply modify the instruction.

.. code:: yaml

    task_name: caption
    instruction:
         - '[IMAGE:image,adaptor=image_vit] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:caption] <EOS>'

