# Copyright 2022 The OFA-Sys Team. All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from enum import Enum


class TaskTemplates(Enum):
    caption = '[IMAGE:img] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:cap] <EOS>'
    cola = (
        '<BOS> is the text " [TEXT:sentence] " grammatically correct? <EOS> -> '
        '<BOS> is the text " [TEXT:sentence,no_loss] " grammatically correct? <EOS> [TEXT:label,closed_set]'
    )
    mnli = (
        '<BOS> can text1 " [TEXT:sent1] " imply text2 " [TEXT:sent2] "? <EOS> -> '
        '<BOS> can text1 " [TEXT:sent1,no_loss] " imply text2 " [TEXT:sent2,no_loss] "? <EOS> [TEXT:label,closed_set]'
    )
    mrpc = (
        '<BOS> does text1 " [TEXT:sent1] " and text2 " [TEXT:sent2] " have the same semantics? <EOS> -> '
        '<BOS> does text1 " [TEXT:sent1,no_loss] " and text2 " [TEXT:sent2,no_loss] " have the same semantics? <EOS> [TEXT:label,closed_set]'
    )
    qnli = (
        '<BOS> does " [TEXT:sent] " contain the answer to question " [TEXT:ques] "? <EOS> -> '
        '<BOS> does " [TEXT:sent,no_loss] " contain the answer to question " [TEXT:ques,no_loss] "? <EOS> [TEXT:label,closed_set]'
    )
    qqp = (
        '<BOS> is question " [TEXT:ques1] " and question " [TEXT:ques2] " equivalent? <EOS> -> '
        '<BOS> is question " [TEXT:ques1,no_loss] " and question " [TEXT:ques2,no_loss] " equivalent? <EOS> [TEXT:label,closed_set]'
    )
    rte = (
        '<BOS> can text1 " [TEXT:sent1] " imply text2 " [TEXT:sent2] "? <EOS> -> '
        '<BOS> can text1 " [TEXT:sent1,no_loss] " imply text2 " [TEXT:sent2,no_loss] "? <EOS> [TEXT:label,closed_set]'
    )
    sst2 = (
        '<BOS> is the sentiment of text " [TEXT:sentence] " positive or negative? <EOS> -> '
        '<BOS> is the sentiment of text " [TEXT:sentence,no_loss] " positive or negative? <EOS> [TEXT:label,closed_set]'
    )
    snli_ve = (
        '[IMAGE:img] <BOS> can image and text1 " [TEXT:cap] " imply text2 " [TEXT:hyp] "? <EOS> -> '
        '<BOS> can image and text1 " [TEXT:cap,no_loss] " imply text2 " [TEXT:hyp,no_loss] "? [TEXT:label,closed_set] <EOS>'
    )
    gigaword = '<BOS> what is the summary of article " [TEXT:src] "? <EOS> -> <BOS> [TEXT:tgt,noise_ratio=0.2] <EOS>'
    refcoco = '[IMAGE:img] <BOS> which region does the text " [TEXT:cap] " describe? <EOS> -> [BOX:region_coord,add_bos,add_eos]'
    image_gen = (
        '<BOS> what is the complete image? caption: [TEXT:text] <EOS> -> '
        '[IMAGE:code,preprocess=image_vqgan,adaptor=image_vqgan,add_bos,add_eos]'
    )
    image_classify = (
        '[IMAGE:image] <BOS> what does the image describe? <EOS> -> <BOS> [TEXT:label_name,closed_set] <EOS>'
    )
    vqa_gen = '[IMAGE:image] <BOS> [TEXT:question] <EOS> -> <BOS> [TEXT:answer,closed_set] <EOS>'
    text_infilling = (
        '<BOS> what is the complete text of " [TEXT:text,mask_ratio=0.3,max_length=256] "? <EOS> -> '
        '<BOS> [TEXT:text,max_length=256] <EOS>'
    )
    image_infilling = (
        '<BOS> what is the complete image of " [IMAGE:img,mask_ratio=0.5] "? <EOS> -> '
        '[IMAGE:code,preprocess=image_vqgan,adaptor=image_vqgan,add_bos,add_eos]'
    )
    image_text_matching = '[IMAGE] <BOS> does the text " [TEXT] " describe the image ?<EOS> -> <BOS> [TEXT] <EOS>'
    asr = '[AUDIO:wav] -> <BOS> [TEXT:text] <EOS>'
    tts = '<BOS> [PHONE:phone] <EOS> -> [AUDIO:fbank,adaptor=audio_tgt_fbank]'
    spider = (
        '<BOS> " [TEXT:src] " ; structured knowledge: " [TEXT:database,max_length=876] " . generating sql code. <EOS> '
        '-> <BOS> [TEXT:tgt,noise_ratio=0.2] <EOS>'
    )
    dart = (
        '<BOS> structured knowledge: " [TEXT:database] " . how to describe the tripleset ? <EOS> -> '
        '<BOS> [TEXT:tgt,noise_ratio=0.2] <EOS>'
    )
    fetaqa = (
        '<BOS> structured knowledge: " [TEXT:database,max_length=768] " . what is the answer of the question " '
        '[TEXT:src,max_length=128] " ? <EOS> -> <BOS> [TEXT:tgt,noise_ratio=0.2] <EOS>'
    )
    sudoku = '<BOS> " [TEXT:src] " . solve the sudoku . <EOS> -> <BOS> [TEXT:tgt] <EOS>'
    natural_instruction_v2 = (
        '<BOS> [TEXT:src,max_length=512]  <EOS> -> <BOS> [TEXT:tgt,noise_ratio=0.2,max_length=128] <EOS>'
    )
    motion_diffusion = (
        '<BOS> human motion: [TEXT:text,max_length=64] <EOS> -> '
        '[MOTION:bvh_frames,preprocess=motion_6d,adaptor=motion_6d,max_length=1024,sample_rate=1]'
    )

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"{value} is not a valid `task`, please select one of {list(cls._value2member_map_.keys())}")
