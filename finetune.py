import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

try:
    from .utils import ScanDataset, LabelSmoothingLoss, Lang, calculate_accuracy
    from .optim_utils import get_inverse_sqrt_schedule_with_warmup
except ImportError:
    from utils import ScanDataset, LabelSmoothingLoss, Lang, calculate_accuracy
    from optim_utils import get_inverse_sqrt_schedule_with_warmup

logger = logging.getLogger(__name__)

SCAN_INPUT_TOKENS_SCAN = ['jump', 'opposite', 'right', 'twice', 'and', 'turn', 'thrice', 'run', 'after', 'around', 'left', 'walk', 'look']
# primitive: jump, run, walk, look, turn
# direction: right, left
# twice, thrice
# prep: and, after
# around, opposite
SCAN_OUTPUT_TOKENS_SCAN = ['I_TURN_RIGHT', 'I_JUMP', 'I_TURN_LEFT', 'I_RUN', 'I_WALK', 'I_LOOK']


class ScanTrainer(BaseTransformer):

    # mode = "language-modeling"

    def __init__(self, hparams):
        config_kwargs = {}

        # if 'dualemb-countaction' in hparams.model_name_or_path:
        #     self.mode = "language-modeling-dualembv0-countaction"
        # elif 'dualemb-v0' in hparams.model_name_or_path:
        self.mode = hparams.model_name_or_path

        config_kwargs: dict = dict(
            mask_decoder_input=hparams.mask_decoder_input,
            embedding_regularization=hparams.embedding_regularization_coef > 0,
            layer_regularization=hparams.layer_regularization_coef > 0,
            predict_action_count=hparams.predict_action_count,
            predict_action_group=hparams.predict_action_group,
            predict_action_count_mode=hparams.predict_action_count_mode,
            action_count_attention_kv=hparams.action_count_attention_kv,
        )
        super().__init__(hparams, num_labels=None, mode=self.mode, **config_kwargs)
        # The tokenizer is initialized in BaseTransformer from AutoTokenizer, but shortcut it here to
        # avoid making our tokenizer AutoTokenizer capable.

        if self.hparams.dataset_name == 'scan':
            self.src_lang = Lang(SCAN_INPUT_TOKENS_SCAN, io_type='input')
            self.trg_lang = Lang(SCAN_OUTPUT_TOKENS_SCAN, io_type='output')
        else:
            raise NotImplementedError

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            sub_task=self.hparams.train_task,
            # train_task=self.hparams.train_task,
            # eval_task=self.hparams.eval_task,
        )

        if self.hparams.label_smooth > 0:
            self.label_smoothing = LabelSmoothingLoss(self.hparams.label_smooth,
                                                      self.config.trg_vocab_size,
                                                      ignore_index=self.trg_lang.pad_token_id)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None,
                action_count_labels=None, action_group_labels=None):
        if self.hparams.predict_action_count or self.hparams.predict_action_group:
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                lm_labels=lm_labels,
                action_count_labels=action_count_labels,
                action_group_labels=action_group_labels,
            )
        else:
            return self.model(
                input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels
            )

    def _step(self, batch, data_type):
        pad_token_id = self.trg_lang.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]

        if self.hparams.dont_prepend_bos:
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        else:
            lm_labels = y.clone()
            lm_labels[y[:, :] == pad_token_id] = -100
            if self.hparams.predict_action_count or self.hparams.predict_action_group:
                if self.hparams.dataset_name == 'scan':
                    action_count_labels, action_group_labels = batch["action_count_labels"], batch["action_group_labels"]
                    action_type_labels = batch["action_type_labels"]

                outputs = self(
                        source_ids,
                        attention_mask=source_mask,
                        lm_labels=lm_labels,
                        action_count_labels=action_count_labels if self.hparams.predict_action_count else None,
                        action_group_labels=action_group_labels if self.hparams.predict_action_group else None,
                    )
            else:
                outputs = self(source_ids, attention_mask=source_mask, lm_labels=lm_labels)

        if self.hparams.label_smooth > 0:
            logits = outputs[1]
            loss = self.label_smoothing(logits, y)
        else:
            loss = outputs[0]

        losses = {}
        overall_loss = loss.clone()

        if self.hparams.predict_action_count:
            action_count_loss = self.model.action_count_loss
            overall_loss += self.hparams.action_count_loss_coeff * action_count_loss
            losses[data_type + '_action_count_loss'] = action_count_loss.clone()
            if self.hparams.count_embedding_regularization_coef > 0: # Do extra regularization for action count/group emb
                count_emb_reg_loss = self.model.count_emb_reg_loss
                overall_loss += self.hparams.count_embedding_regularization_coef * count_emb_reg_loss
                losses[data_type + '_action_count_embedding_regularization_loss'] = count_emb_reg_loss.clone()

        if self.hparams.predict_action_group:
            action_group_loss = self.model.action_group_loss
            overall_loss += self.hparams.action_group_loss_coeff * action_group_loss
            losses[data_type + '_action_group_loss'] = action_group_loss.clone()

        if self.hparams.embedding_regularization_coef > 0:
            emb_reg_loss = self.model.emb_reg_loss
            overall_loss += self.hparams.embedding_regularization_coef * emb_reg_loss
            losses[data_type + '_embedding_regularization_loss'] = emb_reg_loss.clone()

        if self.hparams.layer_regularization_coef > 0:
            layer_reg_loss = self.model.layer_reg_loss
            overall_loss += self.hparams.layer_regularization_coef * layer_reg_loss
            losses[data_type + '_layer_regularization_loss'] = layer_reg_loss.clone()

        losses[data_type + '_cross_entropy_loss'] = loss.clone()
        losses[data_type + '_loss'] = overall_loss.clone()

        return overall_loss, losses

    def training_step(self, batch, batch_idx):
        loss, tensorboard_logs = self._step(batch, data_type='train')
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        pad_token_id = self.trg_lang.pad_token_id
        source_ids, source_mask, y = ScanDataset.trim_seq2seq_batch(
            batch,
            self.src_lang.pad_token_id,
            self.trg_lang.pad_token_id,
            trim_y=True,
        )

        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=80,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
        )

        loss, tensorboard_logs = self._step(batch, data_type='val')

        tensorboard_logs["preds"] = generated_ids
        tensorboard_logs["target"] = y

        return tensorboard_logs

    def _validation_end(self, outputs):
        tensorboard_logs = {}
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_cross_entropy_loss = torch.stack([x["val_cross_entropy_loss"] for x in outputs]).mean()
        tensorboard_logs["val_loss"] = avg_loss
        tensorboard_logs["val_cross_entropy_loss"] = avg_cross_entropy_loss

        if self.hparams.predict_action_count:
            avg_action_count_loss = torch.stack([x["val_action_count_loss"] for x in outputs]).mean()
            tensorboard_logs["val_action_count_loss"] = avg_action_count_loss

        if self.hparams.predict_action_group:
            avg_action_group_loss = torch.stack([x["val_action_group_loss"] for x in outputs]).mean()
            tensorboard_logs["val_action_group_loss"] = avg_action_group_loss

        if self.hparams.embedding_regularization_coef > 0:
            avg_emb_reg_loss = torch.stack([x["val_embedding_regularization_loss"] for x in outputs]).mean()
            tensorboard_logs["val_embedding_regularization_loss"] = avg_emb_reg_loss

        if self.hparams.predict_action_count and self.hparams.count_embedding_regularization_coef > 0:
            avg_count_emb_reg_loss = torch.stack([x["val_action_count_embedding_regularization_loss"] for x in outputs]).mean()
            tensorboard_logs["val_action_count_embedding_regularization_loss"] = avg_count_emb_reg_loss

        if self.hparams.layer_regularization_coef > 0:
            avg_layer_reg_loss = torch.stack([x["val_layer_regularization_loss"] for x in outputs]).mean()
            tensorboard_logs["val_layer_regularization_loss"] = avg_layer_reg_loss

        acc = torch.stack([x["acc"] * x["dp_total"] for x in outputs]).sum() / torch.stack([x["dp_total"] for x in outputs]).sum()
        tensorboard_logs["acc"] = acc
        return {"avg_val_loss": avg_loss, "acc": acc, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+") as p_writer, open(output_test_targets_file, "w+") as t_writer:
            for output_batch in outputs:
                predictions = [
                    self.trg_lang.decode(g) for g in output_batch["preds"]
                ]
                gts = [
                    self.trg_lang.decode(g) for g in output_batch["target"]
                ]

                p_writer.writelines(','.join(s) + "\n" for s in predictions)
                t_writer.writelines(','.join(s) + "\n" for s in gts)

                acc = calculate_accuracy(predictions, gts)
                output_batch['acc'] = torch.tensor(acc)
                output_batch['dp_total'] = torch.tensor(len(gts))

            p_writer.close()
            t_writer.close()

        return self._validation_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _test_end(self, outputs):
        return self._validation_end(outputs)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        if self.hparams.dataset_name == 'scan':
            dataset = ScanDataset(src_lang=self.src_lang, trg_lang=self.trg_lang, type_path=type_path,
                                **self.dataset_kwargs)

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        dataset_total_len = dataloader.num_total_examples if self.hparams.dataloader == 'multifile' else len(dataloader.dataset)
        t_total = (
            (dataset_total_len // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        if self.hparams.scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total,
                last_epoch=self.hparams.resume_from_step,
            )
        elif self.hparams.scheduler_type == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps,
                last_epoch=self.hparams.resume_from_step,
            )
        elif self.hparams.scheduler_type == 'inverse_sqrt':
            scheduler = get_inverse_sqrt_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps,
                last_epoch=self.hparams.resume_from_step,
            )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        # Add BART specific options
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--dataset_name",
            default="cnn_dm",
            type=str,
            help="The data to train/evaluate on.",
        )
        parser.add_argument(
            "--train_task",
            default="addprim_jump",
            type=str,
            help="The data to train/evaluate on.",
        )
        parser.add_argument(
            "--eval_task",
            default="addprim_jump",
            type=str,
            help="The data to evaluate on.",
        )
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )
        parser.add_argument(
            "--run_id",
            default='00',
            type=str,
            help="",
        )
        parser.add_argument(
            "--train_from_scratch",
            action="store_true",
            help="Reinitialize all model weights.",
        )
        parser.add_argument(
            "--overwrite_output_dir",
            action="store_true",
            help="",
        )
        parser.add_argument(
            "--label_smooth",
            default=0.,
            type=float,
        )
        parser.add_argument(
            "--resume_from_epoch",
            default=0,
            type=int,
            help="The checkpoint to restore from."
        )
        parser.add_argument(
            "--resume_from_step",
            default=-1,
            type=int,
            help="The checkpoint to restore from."
        )
        parser.add_argument(
            "--resume_ckpt_path",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--dataloader",
            default="regular",
            type=str,
            help="[regular | multifile | jit]"
        )
        parser.add_argument(
            "--dont_prepend_bos",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--mask_decoder_input",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--embedding_regularization_coef",
            default=0,
            type=float
        )
        parser.add_argument(
            "--layer_regularization_coef",
            default=0,
            type=float
        )
        parser.add_argument(
            "--predict_action_count",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--predict_action_group",
            action="store_true",
            help=""
        )
        parser.add_argument(
            "--action_count_loss_coeff",
            default=1.0,
            type=float
        )
        parser.add_argument(
            "--action_group_loss_coeff",
            default=0.5,
            type=float
        )
        parser.add_argument(
            "--predict_action_count_mode",
            default='MT',
            type=str
        )
        parser.add_argument(
            "--action_count_attention_kv",
            type=str,
            default="f,h",
            help="The key and value input of the count_output_attn."
        )
        parser.add_argument(
            "--count_embedding_regularization_coef",
            default=0,
            type=float
        )

        return parser


def main(args):
    args.data_dir = os.path.join(args.data_dir, args.dataset_name)
    args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.run_id, args.train_task)
    print(args.output_dir)
    if args.resume_from_epoch > 0:
        args.resume_ckpt_path = os.path.join(args.resume_ckpt_path, args.dataset_name, args.run_id, args.train_task)

        checkpoints = list(sorted(
            glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"),
                      recursive=True), key=os.path.getmtime))
        if len(checkpoints) == 0:
            checkpoints = list(sorted(
                glob.glob(os.path.join(args.resume_ckpt_path, "checkpointepoch=" + str(args.resume_from_epoch) + ".ckpt"),
                          recursive=True)))
        args.resume_ckpt_path = checkpoints[-1]
        args.resume_from_step = torch.load(args.resume_ckpt_path)["global_step"]
        print("Resuming from checkpoint:")
        print(args.resume_ckpt_path)
        print("Resuming from step:")
        print(args.resume_from_step)
    else:
        checkpoints = list(sorted(
            glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"),
                      recursive=True), key=os.path.getmtime))
        if len(checkpoints) > 0:
            args.resume_ckpt_path = checkpoints[-1]
            args.resume_from_step = torch.load(args.resume_ckpt_path)["global_step"]
            print("Resuming from checkpoint:")
            print(args.resume_ckpt_path)
            print("Resuming from step:")
            print(args.resume_from_step)

    # If output_dir not provided, a folder will be generated in pwd
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = ScanTrainer(args)
    '''
        # If you want to perform a tokenization run to pre-tokenize the JIT data loader,
        # uncomment this section. The code will fail out at the end of model.train_dataloader()
        # but the tokenization will be completed.
        model.val_dataloader()
        model.test_dataloader()
        model.train_dataloader()
        import sys
        sys.exit()
    '''
    trainer = generic_train(model, args)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        # See https://github.com/huggingface/transformers/issues/3159
        # pl use this format to create a checkpoint:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
        # /pytorch_lightning/callbacks/model_checkpoint.py#L169
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = ScanTrainer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    main(args)