"""
Decode summarization models (generate summaries) trained with finetune.py
Multi-GPU decoding not working yet.
"""

import sys
import argparse
import os
import logging
import glob
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from models.configuration_auxseq import T5Config

from models.modeling_auxseq import T5ForConditionalGenerationWithSepVocab, T5ForConditionalGenerationDualEmb,\
    T5ForConditionalGenerationDualEmbCntAction


from utils import ScanDataset, Lang, calculate_accuracy

logger = logging.getLogger(__name__)

INPUT_TOKENS_SCAN = ['jump', 'opposite', 'right', 'twice', 'and', 'turn', 'thrice', 'run', 'after', 'around', 'left', 'walk', 'look']
OUTPUT_TOKENS_SCAN = ['I_TURN_RIGHT', 'I_JUMP', 'I_TURN_LEFT', 'I_RUN', 'I_WALK', 'I_LOOK']

MODELS = {
    "t5-small": T5ForConditionalGenerationWithSepVocab,
    "t5-small-dualemb": T5ForConditionalGenerationDualEmb,
    "t5-small-dualemb-countaction": T5ForConditionalGenerationDualEmbCntAction,
}

CONFIGS = {
    "t5": T5Config,
}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def decode_scan(args):
    args.batch_size = args.batch_size * max(1, args.n_gpu)

    logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
    checkpoints = list(sorted(glob.glob(os.path.join(args.model_path, "checkpointepoch="+str(args.evaluate_epoch)+".ckpt"), recursive=True)))
    print(os.path.join(args.model_path, "checkpointepoch="+str(args.evaluate_epoch)+".ckpt"))
    checkpoint = checkpoints[0]

    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    num_epoch = checkpoint.split("epoch=")[1].split(".ckpt")[0]

    args.output_path = os.path.join(args.output_path, 'epoch=' + num_epoch + '_beam=' + str(args.num_beams) + '_' + args.output_filename)

    output_file = Path(args.output_path).open("w", encoding='utf-8')

    # Reload the model
    config = CONFIGS[args.model_name_or_path[:2]].from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    config.mask_decoder_input = args.mask_decoder_input
    config.predict_action_count = args.predict_action_count
    config.predict_action_group = args.predict_action_group
    config.predict_action_count_mode = args.predict_action_count_mode
    config.action_count_attention_kv = args.action_count_attention_kv

    # mode = args.model_name_or_path[:2]
    #
    # if 'dualembv0-countaction' in args.model_name_or_path:
    #     mode += '-dualemb-countaction'
    # elif 'dualemb-v0' in args.model_name_or_path:
    #     mode += '-dualemb'

    model = MODELS[args.model_name_or_path].from_pretrained(
        args.model_name_or_path,
        return_unpretrained=True,
        config=config,
        cache_dir=args.cache_dir,
    )

    # Restore the model parameters
    state_dict_pl = torch.load(checkpoint)["state_dict"]
    state_dict = {}
    for weight in state_dict_pl:
        if 'label_smoothing' in weight or 'rq' in weight or 'rk' in weight:
            continue
        if 'action_count_head' in weight and not args.predict_action_count:
            continue
        if weight.startswith("model."):
            state_dict[weight[6:]] = state_dict_pl[weight]
        else:
            state_dict[weight] = state_dict_pl[weight]

    model.load_state_dict(state_dict)
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.dataset_name == 'scan':
        src_lang = Lang(INPUT_TOKENS_SCAN, io_type='input')
        trg_lang = Lang(OUTPUT_TOKENS_SCAN, io_type='output')
        dataset = ScanDataset(
            src_lang=src_lang,
            trg_lang=trg_lang,
            type_path=args.data_split,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            sub_task=args.eval_task,
        )
    else:
        raise NotImplementedError

    eval_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    epoch_iterator = tqdm(dataloader, desc="Iteration", disable=False)
    correct_count, action_count_correct, action_group_correct = 0, 0, 0
    short_cnt, short_correct_cnt = 0, 0
    long_opposite_cnt, long_opposite_correct_cnt, long_around_cnt, long_around_correct_cnt = 0, 0, 0, 0
    long_oppo_only_cnt, long_oppo_only_correct_cnt, long_around_only_cnt, long_around_only_correct_cnt = 0, 0, 0, 0

    for step, batch in enumerate(epoch_iterator):
        model.eval()
        input_ids = batch["source_ids"].to(args.device)
        attention_mask = batch["source_mask"].to(args.device)
        gt_action_count_ids, gt_action_group_ids = None, None
        if args.predict_action_count:
            if args.use_gt_action_count_ids:
                gt_action_count_ids = batch["action_count_labels"]

        if args.predict_action_group:
            if args.use_gt_action_group_ids:
                gt_action_group_ids = batch["action_group_labels"]

        summaries = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True,
            predict_action_count=args.predict_action_count,
            gt_action_count_ids=gt_action_count_ids,
            gt_action_group_ids=gt_action_group_ids,
        )

        inputs = [src_lang.decode(g) for g in batch["source_ids"]]
        dec = [trg_lang.decode(g) for g in summaries]
        gts = [trg_lang.decode(g) for g in batch["target_ids"]]

        # dec_segs = dec
        # dec = trg_lang.merge_sequences(dec)
        # gts = trg_lang.merge_sequences(gts + seg2_gts)

        for ih, hypothesis in enumerate(dec):
            input, gt = inputs[ih], gts[ih]
            output_file.write(' '.join(input) + "\n")
            output_file.write(' '.join(hypothesis) + "\n")
            output_file.write(' '.join(gt) + "\n\n")
            output_file.flush()
            input_text, hyp_text, gt_text = ' '.join(input), ' '.join(hypothesis), ' '.join(gts[ih])

            if args.predict_action_count:
                action_count_pred = list(model.action_count_pred[ih].cpu().numpy())
                action_count_pred = action_count_pred[:len(hypothesis)]
                # action_count_pred = action_count_pred[:action_count_pred.index(7)]
                action_count_gold = list(batch["action_count_labels"][ih].cpu().numpy())
                action_count_gold = action_count_gold[:action_count_gold.index(-1)]

                if action_count_pred == action_count_gold:
                    action_count_correct += 1

            if args.predict_action_group:
                action_group_pred = list(model.action_group_pred[ih].cpu().numpy())
                action_group_pred = action_group_pred[:len(hypothesis)]
                action_group_gold = list(batch["action_group_labels"][ih].cpu().numpy())
                action_group_gold = action_group_gold[:action_group_gold.index(-1)]

                if action_group_pred == action_group_gold:
                    action_group_correct += 1

            if hypothesis == gts[ih]:
                correct_count += 1

            if 'around left twice' not in input_text and 'around right twice' not in input_text \
                and 'opposite left thrice' not in input_text and 'opposite right thrice' not in input_text:
                short_cnt += 1
                if hypothesis == gts[ih]:
                    short_correct_cnt += 1
            else:
                if ('opposite left thrice' in input_text or 'opposite right thrice' in input_text):
                    long_opposite_cnt += 1
                    if hypothesis == gts[ih]:
                        long_opposite_correct_cnt += 1
                    if ('around left twice' not in input_text and 'around right twice' not in input_text):
                        long_oppo_only_cnt += 1
                        if hypothesis == gts[ih]:
                            long_oppo_only_correct_cnt += 1

                if ('around left twice' in input_text or 'around right twice' in input_text):
                    long_around_cnt += 1
                    if hypothesis == gts[ih]:
                        long_around_correct_cnt += 1
                    if ('opposite left thrice' not in input_text and 'opposite right thrice' not in input_text):
                        long_around_only_cnt += 1
                        if hypothesis == gts[ih]:
                            long_around_only_correct_cnt += 1

                if hypothesis != gts[ih]:
                    print(input_text)
                    print(action_count_pred)
                    print(action_group_pred)
                    print(hyp_text)
                    print(gt_text + '\n')

    print(correct_count, len(dataset))
    print("Accuracy: %f" % (correct_count / len(dataset)))
    # print("Accuracy without Around Twice & Opposite Thrice: %f" % (short_correct_cnt / short_cnt))
    # print("Accuracy with Around Twice or Opposite Thrice: %f" % ((correct_count - short_correct_cnt) / (len(dataset) - short_cnt)))
    # print("Accuracy with Around Twice: %f" % (long_around_correct_cnt / long_around_cnt))
    # print("Accuracy with Opposite Thrice: %f" % (long_opposite_correct_cnt / long_opposite_cnt))
    # print("Accuracy with Around Twice but not Opposite Thrice: %f" % (long_around_only_correct_cnt / long_around_only_cnt))
    # print("Accuracy with Opposite Thrice but not Around Twice: %f" % (long_oppo_only_correct_cnt / long_oppo_only_cnt))
    if args.predict_action_count:
        print("Action count accuracy: %f" % (action_count_correct / len(dataset)))
    if args.predict_action_group:
        print("Action group accuracy: %f" % (action_group_correct / len(dataset)))


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="",
    )
    parser.add_argument(
        "--max_source_length", type=int, default=512, help="",
    )
    parser.add_argument(
        "--max_target_length", type=int, default=56, help="",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache", help="",
    )
    parser.add_argument(
        "--model_path", type=str, default="./out", help="the location of the model to be eval.",
    )
    parser.add_argument(
        "--run_id", type=str, default='00', help="",
    )
    parser.add_argument(
        "--evaluate_epoch", type=int, default=-1, help="",
    )
    parser.add_argument(
        "--dataset_name", default="scan", type=str, help="The data to evaluate on.",
    )
    parser.add_argument(
        "--data_split", default="val", type=str, help="The data to evaluate on.",
    )
    parser.add_argument(
        "--eval_task", default="addprim_jump", type=str, help="The Scan subtask to evaluate on.",
    )
    parser.add_argument(
        "--eval_dataset_name", type=str, help="The data to evaluate on.",
    )
    parser.add_argument(
        "--data_dir", default="./data", type=str, help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
    )
    parser.add_argument(
        "--input_path", type=str, default="val.source", help="like cnn_dm/test_articles_input.txt",
    )
    parser.add_argument(
        "--reference_path", type=str, default="val.target", help="like cnn_dm/test_reference_summaries.txt"
    )
    parser.add_argument(
        "--output_path", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "--output_filename", type=str, default="generated_summaries.txt", help="where to save summaries",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, required=False, help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda", default=False, type=bool, help="Whether to force the execution on CPU.",
    )
    parser.add_argument(
        "--num_beams", default=1, type=int, help="Beam size."
    )
    parser.add_argument(
        "--max_length", default=200, type=int, help="The max length of generated summaries."
    )
    parser.add_argument(
        "--min_length", default=1, type=int, help="The min length of generated summaries."
    )
    parser.add_argument(
        "--trained_on_task", default='addprim_jump', type=str, help="The min length of generated summaries."
    )
    parser.add_argument(
        "--mask_decoder_input", action="store_true"
    )
    parser.add_argument(
        "--predict_action_count", action="store_true"
    )
    parser.add_argument(
        "--use_gt_action_count_ids", action="store_true"
    )
    parser.add_argument(
        "--predict_action_group", action="store_true"
    )
    parser.add_argument(
        "--use_gt_action_group_ids", action="store_true"
    )
    parser.add_argument(
        "--predict_action_count_mode", default="MT", type=str
    )
    parser.add_argument(
        "--action_count_attention_kv", default="f,c", type=str
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.eval_dataset_name is None:
        args.eval_dataset_name = args.dataset_name
    args.model_path = os.path.join(args.model_path, args.dataset_name, args.run_id, args.trained_on_task)
    if args.output_path is None:
        args.output_path = os.path.join(args.model_path, args.eval_dataset_name)
    else:
        args.output_path = os.path.join(args.output_path, args.eval_dataset_name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.data_dir = os.path.join(args.data_dir, args.eval_dataset_name)
    args.input_path = os.path.join(args.data_dir, args.input_path)
    args.reference_path = os.path.join(args.data_dir, args.reference_path)

    decode_scan(args)


if __name__ == "__main__":
    run_generate()