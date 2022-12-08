# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import re
import warnings
from typing import Any, Dict

from PIL import Image, ImageFile

from ofasys.configure import register_config
from ofasys.task.base import OFATask, TaskConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


@register_config("ofasys.task", "video_question_answering_generative", dataclass=TaskConfig)
class VideoQuestionAnsweringGenerativeTask(OFATask):
    def pre_question(self, question, max_ques_words=None):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question

    def preprocess(self, data: Dict[str, Any], split: str) -> Dict[str, Any]:

        question = self.pre_question(data['question'], self.cfg.max_src_length)
        question = question + '?' if not question.endswith('?') else question

        assert '&&' not in data['answer']
        ref_dict = {item: 1.0 for item in data['answer'].split('&&')}
        answer = max(ref_dict, key=ref_dict.get)
        conf = ref_dict[answer]
        #
        if data.get('predict_objects', None) is not None:
            predict_object_seq = ' '.join(data['predict_objects'].strip().split('&&')[: self.cfg.max_object_length])
            predict_object_seq = " object: {}".format(predict_object_seq)
            question = predict_object_seq + f' {question}'

        data["question"] = question
        data["answer"] = answer
        data["conf"] = conf
        data["ref_dict"] = ref_dict

        return data
