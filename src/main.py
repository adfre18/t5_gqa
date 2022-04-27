import copy

from transformers import AutoConfig, T5Tokenizer
from models.t5.modeling_t5_gqa import T5ModelGQA
from src.utils.GQADataset import GQADataset, load_img_features
from src.utils.arg_parser import parser
from src.train_procedure import train
from src.utils.filter_input_data import filter_input_data
import torch
import numpy as np
import h5py

if __name__ == '__main__':
    args = parser.parse_args()
    args.n_gpu = 1
    # args = {'data_dir': 'D:\Dokumenty\PhD\GQA\GitRep\data'}
    img_features = load_img_features(args)
    label_pos_feats = None
    t5_config = AutoConfig.from_pretrained(
        'config.json'
    )
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    tokenizer.add_tokens("<sep>", special_tokens=True)
    tokenizer.sep_token = "<sep>"

    model = T5ModelGQA.from_pretrained('t5-base', config=t5_config)
    model.resize_token_embeddings(len(tokenizer))

    with h5py.File('C:\GitRepo\T5_GQA\data\coco_and_gqa_test_box_features_ext.hdf5', "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        dict_with_examples = {}
        for idx, key in enumerate(list(f.keys())):
            if idx < 1000:
                data_raw = torch.tensor(np.array(f[key]))
                data = np.array(f[key])
                dict_with_examples[key] = data
            else:
                break

    if args.do_train:
        train_dataset = GQADataset(args, 'train', img_features, tokenizer, label_pos_feats)
        filter_input_data(train_dataset, img_features)
        #captions = GQADataset(args, 'my_test', img_features, tokenizer, label_pos_feats)
        eval_dataset = GQADataset(args, 'val', img_features, tokenizer, label_pos_feats)
        filter_input_data(eval_dataset, img_features)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    t5_model = T5ModelGQA(t5_config)
    print()
    #img_features = torch.load('D:\Dokumenty\PhD\GQA\GitRep\data\\extracted_feats.pt')
    with h5py.File('C:\GitRepo\T5_GQA\data\coco_and_gqa_test_box_features_ext.hdf5', "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        dict_with_examples = {}
        for idx, key in enumerate(list(f.keys())):
            if idx < 1000:
                data_raw = torch.tensor(np.array(f[key]))
                data = np.array(f[key])
                dict_with_examples[key] = data
            else:
                break
        # Get the data
        #data = list(f[a_group_key])
    print()