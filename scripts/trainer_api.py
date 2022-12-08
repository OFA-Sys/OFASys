from ofasys import Task, Trainer, GeneralistModel

# 1. Define the multi-modal tasks
task1 = Task(
    name='caption',
    instruction='[IMAGE:image_url] what does the image describe? -> [TEXT:caption]',
    micro_batch_size=4,
)
task2 = Task(
    name='text_infilling',
    instruction='what is the complete text of " [TEXT:sentence,mask_ratio=0.3] "? -> [TEXT:sentence]',
    micro_batch_size=2,
)

# 2. Bind the dataset
from datasets import load_dataset

task1.add_dataset(load_dataset('TheFusion21/PokemonCards')['train'], 'train')
task2.add_dataset(load_dataset('glue', 'cola')['train'], 'train')

# 3. Create an OFA-Sys Unify Model
model = GeneralistModel()
# model.cfg.arch = 'base'

# 4. Train all tasks together
trainer = Trainer()
trainer.fit(model=model, tasks=[task1, task2])
