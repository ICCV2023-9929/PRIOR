from nltk import text
from numpy import log
import sys
import os
sys.path.append(os.getcwd())
from transformers import AutoTokenizer
from prior.data.pretrain.base import PretrainDataset
from prior.data.pretrain.text_process import SentenceSplitter, split_into_sections
from prior.transforms.language import SentenceShuffle
import matplotlib.pyplot as plt
import pydicom
import logging
import json
import click
from PIL import Image, ImageFile
import glob
from torchvision import transforms
import torch
import random
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer


log = logging.getLogger(__name__)
logging.getLogger('stanfordcorenlp').setLevel(logging.WARNING)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MimicCxrDataset(PretrainDataset):
    def __init__(self, dataset_path, num_colors=1, image_transform=[], text_transform=[], rate=1.0,  max_length=256):
        self.rate = rate
        self.num_colors = num_colors
        self.filter = filter
        super().__init__(dataset_path, image_transform=image_transform, text_transform=text_transform, rate=rate)
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        if self.num_colors == 1:
            self.image_io = Image.open
        else:
            self.image_io = lambda x: Image.open(x).convert('RGB')
        self.max_length = max_length

    def _load_text_dataset(self):
        with open(self.dataset_path, 'r') as f:
            self.data_json = json.load(f)
        self.id_list = list(self.data_json.keys())
        
    def _load_image_dataset(self):
        pass

    def _build_transform(self):
        self.image_transform += [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        self.image_transform = transforms.Compose(self.image_transform)

           
    def _get_image(self, index):
        images = []
        for image_path in self.data_json[self.id_list[index]]['image_path']:
            item = self.image_io(image_path)
            images.append(item)
        return images[random.randint(0, len(images)-1)] # random choose one image
    
    def _get_text(self, index):
        text = self.data_json[self.id_list[index]]['text_info']
        return self._tokenizer(text, index)

    def _tokenizer(self, text, index): 
        sentences_list = []
        sentences = ''
        if 'impression' in text.keys() and len(text['impression'][1]) != 0:
            for idx, sentence in enumerate(text['impression'][1]):
                sentences += sentence
                sentences_list.append(sentence)
        if 'findings' in text.keys() and len(text['findings'][1]) != 0:
            for idx, sentence in enumerate(text['findings'][1]):
                sentences += sentence  
                sentences_list.append(sentence)
        sentences = SentenceShuffle()(sentences_list) # randome shuffling setences
        tokens = self.tokenizer(
        sentences,
        max_length=self.max_length,
        add_special_tokens=True,
        padding='max_length',
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )
        for key, token in tokens.items():
            tokens[key] = token.squeeze(dim=0)
        num_tokens = 0
        sentence_index = torch.ones_like(tokens['input_ids']) * -1
        start_index = 1

        # assign the index of sentence to each token
        for idx, sentence in enumerate(sentences_list):
            sentnece_token = self.tokenizer(sentence, return_tensors='pt')['input_ids'][0]
            len_sentence = sentnece_token.shape[0] - 2 # Remove [CLS] and [SEP]
            sentence_index[start_index: start_index + len_sentence] = idx + 1
            start_index += len_sentence

        # truncation 
        for key, value in tokens.items():
            #print(len(value))
            tokens[key] = value
            #print(len(tokens[key]))
            if len(value) > self.max_length:
                tokens[key] = value[ :self.max_length]
        if len(sentence_index) > self.max_length:
            sentence_index = sentence_index[ :self.max_length]
    
        return tokens, {'num_tokens': num_tokens, 'sentence_index': sentence_index} 
 
    def _load_statics(self):
        if self.num_colors == 1: 
            self.mean = [0.4755]
            self.std = [0.3011]
        else:
            self.mean = [0.4755, 0.4755, 0.406] # ImageNet 
            self.std = [0.229, 0.224, 0.225]

    def _get_meta(self, index):
        return np.asanyarray(self.data_json[self.id_list[index]]['label']).astype(np.int64)


    def __getitem__(self, index):
        ori_image = self._get_image(index)
        image = self.image_transform(ori_image)
        label = self._get_meta(index)
        text, text_meta = self._get_text(index)
        batch = {'image': image, 'text': text, 'label': label}
        if text_meta is not None:
            batch['text_meta'] = text_meta
        return batch
    
    def __len__(self):
        if self.rate < 1:
            return int(len(self.id_list) * self.rate)
        return len(self.id_list) 


# ===========================================================================================

'''
======================
======= Builder ======
======================
'''
class MimicCxrDatasetBuilder():
    def __init__(self, root_image_path, root_text_path, save_path, min_sentences=4, mode='jpg'):
        self.root_image_path = root_image_path
        self.save_path = save_path
        self.root_text_path = root_text_path
        self.min_sentences = min_sentences
        self.tokenizer = SentenceSplitter(lang='en', section_splitter=split_into_sections)
        os.makedirs(save_path, exist_ok=True)
        if mode == 'dicom':
            self.ext = 'dcm'
            self.io_read =  pydicom.read_file
        elif mode == 'jpg':
            self.ext = 'jpg'
            self.io_read = plt.imread
        self.sub_part = ['p{}'.format(str(idx)) for idx in range(10, 20)]
        
    def build(self):
        data_json = {}
        for p_index in os.listdir(self.root_image_path):
            print(p_index)
            p_path = os.path.join(self.root_image_path, p_index)
            for patient_id in os.listdir(p_path):
                patient_path = os.path.join(p_path, patient_id)
                for image_id in os.listdir(patient_path):
                    if image_id.split('.')[-1] == 'txt': # Skip text
                        continue
                    image_list = tuple(glob.glob(os.path.join(patient_path, image_id, '*.{}'.format(self.ext))))
                    text_path = os.path.join(self.root_text_path, p_index, patient_id, image_id + '.txt')
                    if not os.path.exists(text_path):
                        log.info('[Sample {} is missing]'.format(image_id))
                    with open(text_path, encoding='utf-8') as f:
                        text = f.read()
                    text = self.tokenizer(text, study=f's{image_id}')
                    #assert image_id not in data_json.keys()
                    if text is None:
                        log.info('[WARNING] {} cannot be tokenized'.format(image_id))
                        continue
                    sentences = [] 
                    total_sentences = ''
                    if 'impression' in text.keys() and len(text['impression'][1]) != 0:
                        for idx, sentence in enumerate(text['impression'][1]):
                            sentences.append(sentence)
                            total_sentences += sentence
                    if 'findings' in text.keys() and len(text['findings'][1]) != 0:
                        for idx, sentence in enumerate(text['findings'][1]):
                            sentences.append(sentence)
                            total_sentences += sentence
                    if len(sentences) < self.min_sentences:
                        log.info('[WARNING] {}\'s number of sentences is {}, which cannot achieve {}'.format(image_id, len(sentences), self.min_sentences))
                        continue
                    data_json[image_id] = {}
                    data_json[image_id]['image_path'] = image_list
                    data_json[image_id]['p_index'] = p_index
                    data_json[image_id]['text_info'] = text
                    data_json[image_id]['patient_id'] = patient_id
                    data_json[image_id]['total_sentences'] = total_sentences
                    data_json[image_id]['sentence_number'] = len(sentences)
        with open(os.path.join(self.save_path, 'data.json'), 'w') as f:
            json.dump(data_json, f)


# ===========================================================================================

'''
======================
======== Debug =======
======================
'''


@click.command('build')
@click.option('--root_image_path', help='directory of MIMIC-CXR image dataset')
@click.option('--root_text_path', help='directory of MIMIC-CXR image dataset')
@click.option('--save_path', help='path of saved file')
@click.option('--mode',  default='jpg', help='path of saved file')
def build_mimic_cxr_dataset_with_split(root_image_path, root_text_path, save_path, mode):
    builder = MimicCxrDatasetBuilder(root_image_path=root_image_path, root_text_path=root_text_path, save_path=save_path, mode=mode)
    builder.build()

@click.group()
def cli():
    pass

cli.add_command(build_mimic_cxr_dataset_with_split)

if __name__ == '__main__':
    cli()



    


