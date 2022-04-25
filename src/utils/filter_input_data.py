from .task_utils import processors, output_modes, _truncate_seq_pair, convert_examples_to_features_vqa


def filter_input_data(dataset, img_feats):
    filtered_examples = list()
    for example in dataset.examples:
        if example.img_key in img_feats.keys():
            filtered_examples.append(example)
    dataset.examples = filtered_examples