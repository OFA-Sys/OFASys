import os

import torch
from modelscope.models.base import Model, TorchModel
from modelscope.models.builder import MODELS
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import ModelFile


@MODELS.register_module('my-ofasys-task', module_name='my-custom-model')
class MaasTemplateModel(TorchModel):
    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model = self.init_model(**kwargs)

    def forward(self, input_tensor, **forward_params):
        if forward_params.get('instruction') is None:
            raise ValueError('instruction is missing')
        else:
            template = forward_params.pop('instruction')
        return self.model.inference(template, data=input_tensor, **forward_params)

    def init_model(self, **kwargs):
        from ofasys import OFASys

        model = OFASys.from_pretrained(os.path.join(self.model_dir, ModelFile.TORCH_MODEL_BIN_FILE))
        return model.cuda(0)


@PREPROCESSORS.register_module('my-ofasys-task', module_name='my-custom-preprocessor')
class MaasTemplatePreprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        """image instance segmentation preprocessor in the fine-tune scenario"""
        super().__init__(*args, **kwargs)
        self.trainsforms = self.init_preprocessor(**kwargs)

    def __call__(self, results):
        return self.trainsforms(results)

    def init_preprocessor(self, **kwarg):
        return lambda x: x


@PIPELINES.register_module('my-ofasys-task', module_name='my-custom-pipeline')
class MaasTemplatePipeline(Pipeline):
    def __init__(self, model, preprocessor=None, **kwargs):
        """
        use `model` and `preprocessor` to create a ocr recognition pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False)
        assert isinstance(model, str) or isinstance(
            model, Model
        ), 'model must be a single str or MaasVisionTransformerPipeline'
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.eval()
        if preprocessor is None:
            preprocessor = MaasTemplatePreprocessor()
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output
        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, inputs, **forward_params):
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs):
        return inputs


if __name__ == "__main__":
    from modelscope.pipelines import pipeline

    model = "damo/ofasys_multimodal_multitask_pretrain_base_en"
    pipe = pipeline('my-ofasys-task', model=model)
    instruction = '[IMAGE:img] what does the image describe? -> [TEXT:cap]'
    data = {
        'img': "https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/coco/2014/val2014/COCO_val2014_000000222628.jpg"
    }
    output = pipe(data, instruction=instruction)
    print(output.text)
