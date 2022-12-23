import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from ofasys import OFASys

model = OFASys.from_pretrained('https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/model_hub/multitask_10k.pt')
model = model.cuda(0)


instruction = '[IMAGE:img] what does the image describe? -> [TEXT:cap]'
data = {'img': "https://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/coco/2014/val2014/COCO_val2014_000000222628.jpg"}
output = model.inference(instruction, data=data)
print(output.text)
# a man and woman sitting in front of a laptop computer


instruction = '[IMAGE:img] which region does the text " [TEXT:cap] " describe? -> [BOX:region_coord]'
data = [
    {'img': "https://www.2008php.com/2014_Website_appreciate/2015-06-22/20150622131649.jpg", 'cap': 'hand'},
    {
        'img': "http://ofasys.oss-cn-zhangjiakou.aliyuncs.com/data/coco/2014/train2014/COCO_train2014_000000581563.jpg",
        'cap': 'taxi',
    },
]
output = model.inference(instruction, data=data)
for i, out in enumerate(output):
    out.save_box(f'{i}')


instruction = 'what is the summary of article " [TEXT:src] "? -> [TEXT:tgt]'
data = {
    'src': "poland 's main opposition party tuesday endorsed president lech walesa in an upcoming presidential "
    "run-off election after a reformed communist won the first round of voting ."
}
output = model.inference(instruction, data=data)
print(output.text)
# polish opposition endorses walesa in presidential run-off

instruction = '[IMAGE:img] what does the region describe? region: [BOX:region_coord] -> [TEXT:cap]'
data = {
    'img': 'oss://ofasys/data/coco/2014/train2014/COCO_train2014_000000575980.jpg',
    'region_coord': '32.7,56.07,204.74,292.47',
}
output = model.inference(instruction, data=data)
print(output.text)
# boy on skateboard

instruction = 'structured knowledge: " [STRUCT:database,uncased] "  . how to describe the tripleset ? -> [TEXT:tgt]'
data = {
    'database': [
        ['Atlanta', 'OFFICIAL_POPULATION', '5,457,831'],
        ['[TABLECONTEXT]', 'METROPOLITAN_AREA', 'Atlanta'],
        ['5,457,831', 'YEAR', '2012'],
        ['[TABLECONTEXT]', '[TITLE]', 'List of metropolitan areas by population'],
        ['Atlanta', 'COUNTRY', 'United States'],
    ]
}
output = model.inference(instruction, data=data, beam_size=1)
print(output.text)
# atlanta, united states has a population of 5,457,831 in 2012.


instruction = (
    '" [TEXT:src] " ; structured knowledge: " [STRUCT:database,max_length=876] " . generating sql code. -> [TEXT:tgt]'
)
database = [
    ['concert_singer'],
    ['stadium', 'stadium_id , location , name , capacity , highest , lowest , average'],
    ['singer', 'singer_id , name , country , song_name , song_release_year , age , is_male'],
    ['concert', 'concert_id , concert_name , theme , stadium_id , year'],
    ['singer_in_concert', 'concert_id , singer_id'],
]
data = [
    {
        'src': 'What are the names, countries, and ages for every singer in descending order of age?',
        'database': database,
    },
    {'src': 'What are all distinct countries where singers above age 20 are from?', 'database': database},
    {'src': 'Show the name and the release year of the song by the youngest singer.', 'database': database},
]
output = model.inference(instruction, data=data)
print('\n'.join(o.text for o in output))
# select name, country, age from singer order by age desc
# select distinct country from singer where age > 20
# select song_name, song_release_year from singer order by age limit 1


instruction = '[VIDEO:video] what does the video describe? -> [TEXT:cap]'
data = {'video': 'oss://ofasys/datasets/msrvtt_data/videos/video7021.mp4'}
output = model.inference(instruction, data=data)
print(output.text)
# a baseball player is hitting a ball


instruction = '[AUDIO:wav] what is the text corresponding to the voice? -> [TEXT:text,preprocess=text_phone]'
data = {'wav': 'oss://ofasys/data/librispeech/dev-clean/1272/128104/1272-128104-0001.flac'}
output = model.inference(instruction, data=data)
print(output.text)
# nor is mister klohs manner less interesting than his manner

instruction = (
    'what is the complete image? caption: [TEXT:text]"? -> [IMAGE,preprocess=image_vqgan,adaptor=image_vqgan]'
)
data = {'text': "a city with tall buildings and a large green park."}
output = model.inference(instruction, data=data)
output[0].save_image('0.png')
