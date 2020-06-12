import os
import warnings
from pathlib import Path
import json
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from utils import BERTReduceLROnPlateau
from utils import BertAdam
from utils import ModelCheckpoint
from utils import  EarlyStopping
from utils.tools import logger,seed_everything,init_logger
from configs.base import config
from models.bert_crf import BERTCRF
from modules.bert.bert_seq_processor import BertProcessor
from distill_trainer import DistillTrainer
from ner_seq_trainer import Trainer
from modules.bert.configuration_bert import BertConfig
from modules.bert.bert_seq_processor import convert_data_to_tensor
from utils.logger import Tensorboard_Logger

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def run_train(args):
	processor = BertProcessor (vocab_path=os.path.join (args.pretrained_model, 'vocab.txt',),test_mode=args.test_mode,
	                           do_lower_case=args.do_lower_case)
	#processor.tokenizer.save_vocabulary (str (args.model_path))
	label_list = processor.get_labels ()
	label2id = {label: i for i, label in enumerate (label_list)}
	train_cache_sample = config['data_dir'] / f"cached_train_seq_examples"
	train_cache_feature = config['data_dir'] / f"cached_train_seq_features"
	if args.type:
		train_data = processor.read_type_data (os.path.join (config['data_dir'], "train.jsonl"), type=args.type)
		valid_data = processor.read_type_data (os.path.join (config['data_dir'], "dev.jsonl"), type=args.type)
		train_cache_sample = config['data_dir'] / f"cached_train_seq_examples_{args.type}"
		train_cache_feature = config['data_dir'] / f"cached_train_seq_features_{args.type}"
	else:
		train_data = processor.read_data (os.path.join (config['data_dir'], "train.jsonl"))
		valid_data = processor.read_data (os.path.join (config['data_dir'], "dev.jsonl"))
	if args.early_stop:
		early_stopping = EarlyStopping (patience=3, monitor="f1", baseline=0, mode='max')
	else:
		early_stopping = None

	train_dataset = convert_data_to_tensor (processor=processor,
	                                        args=args,
	                                        data=train_data,
	                                        type="train",
	                                        cache_sample_path=train_cache_sample,
	                                        cache_feature_path=train_cache_feature, save_cache=False)

	if args.sorted:
		train_sampler = SequentialSampler (train_dataset)
	else:
		train_sampler = RandomSampler (train_dataset)
	train_dataloader = DataLoader (train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


	valid_dataset = convert_data_to_tensor (processor=processor,
	                                        args=args,
	                                        data=valid_data,
	                                        type="dev",
	                                        cache_sample_path=config['data_dir'] / f"cached_dev_seq_examples",
	                                        cache_feature_path=config['data_dir'] / f"cached_dev_seq_features",
	                                        save_cache=False)
	valid_sampler = SequentialSampler (valid_dataset)
	valid_dataloader = DataLoader (valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)


	tmodel = BERTCRF
	smodel=  BERTCRF

	t_config = BertConfig.from_json_file(os.path.join(args.pretrained_model,"config.json"))
	s_config = BertConfig.from_json_file(os.path.join(args.pretrained_model,"config.json"))
	s_config.num_hidden_layers = args.depth
	args.tea_resume_path = Path (args.tea_resume_path)
	args.stu_resume_path = Path(args.stu_resume_path)
	tmodel = tmodel.from_pretrained (args.tea_resume_path, label2id=label2id, device=args.device,config=t_config)
	smodel = smodel.from_pretrained(args.stu_resume_path,label2id=label2id, device=args.device,config=s_config)
	tmodel = tmodel.to(args.device)
	smodel = smodel.to(args.device)

	t_total = int (len (train_dataloader) / args.gradient_accumulation_steps * args.epochs)

	bert_param_optimizer = list (smodel.bert.named_parameters ())
	crf_param_optimizer = list (smodel.crf.named_parameters ())
	linear_param_optimizer = list (smodel.classifier.named_parameters ())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in bert_param_optimizer if not any (nd in n for nd in no_decay)], 'weight_decay': 0.01,
		 'lr': args.learning_rate},
		{'params': [p for n, p in bert_param_optimizer if any (nd in n for nd in no_decay)], 'weight_decay': 0.0,
		 'lr': args.learning_rate},
		{'params': [p for n, p in crf_param_optimizer if not any (nd in n for nd in no_decay)], 'weight_decay': 0.01,
		 'lr': args.ex_learning_rate},
		{'params': [p for n, p in crf_param_optimizer if any (nd in n for nd in no_decay)], 'weight_decay': 0.0,
		 'lr': args.ex_learning_rate},
		{'params': [p for n, p in linear_param_optimizer if not any (nd in n for nd in no_decay)], 'weight_decay': 0.01,
		 'lr': args.ex_learning_rate},
		{'params': [p for n, p in linear_param_optimizer if any (nd in n for nd in no_decay)], 'weight_decay': 0.0,
		 'lr': args.ex_learning_rate}
	]
	if args.optimizer == 'adam':
		optimizer = BertAdam (optimizer_grouped_parameters, lr=args.learning_rate,
		                      warmup=args.warmup_proportion, t_total=t_total)
	else:
		raise ValueError ("unknown optimizer")
	lr_scheduler = BERTReduceLROnPlateau (optimizer, lr=args.learning_rate, mode=args.mode, factor=0.5, patience=1,
	                                      verbose=1, epsilon=1e-8, cooldown=0, min_lr=0, eps=1e-8)


	model_checkpoint = ModelCheckpoint (checkpoint_dir=args.model_path, mode=args.mode, monitor=args.monitor,
	                                    arch=args.arch, save_best_only=args.save_best)

	# **************************** training model ***********************
	logger.info ("***** Running training *****")
	logger.info ("  Num Epochs = %d", args.epochs)
	logger.info ("  Total train batch size (w. parallel, distributed & accumulation) = %d",
	             args.train_batch_size * args.gradient_accumulation_steps * (
		             torch.distributed.get_world_size () if args.local_rank != -1 else 1))
	logger.info ("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info ("  Total optimization steps = %d", t_total)
	tb_logger = Tensorboard_Logger (log_dir=os.path.join (args.model_path, 'tb_logger'))

	trainer = DistillTrainer (n_gpu=args.n_gpu,
	                   tmodel=tmodel,
					   smodel=smodel,
	                   logger=logger,
	                   tb_logger=tb_logger,
	                   optimizer=optimizer,
	                   lr_scheduler=lr_scheduler,
	                   label2id=label2id,
	                   grad_clip=args.grad_clip,
	                   model_checkpoint=model_checkpoint,
	                   gradient_accumulation_steps=args.gradient_accumulation_steps,
	                   early_stopping=early_stopping,
	                   partial=args.partial,
	                   trigger=args.trigger,
					   alpha=args.alpha)

	trainer.train (train_data=train_dataloader, valid_data=valid_dataloader, epochs=args.epochs, seed=args.seed)


def run_test(args):
	processor = BertProcessor (os.path.join (args.pretrained_model, 'vocab.txt'), args.do_lower_case,test_mode=args.test_mode)
	label_list = processor.get_labels ()
	label2id = {label: i for i, label in enumerate (label_list)}
	# id2label = {i: label for i, label in enumerate (label_list)}

	model = BERTCRF
	bert_config = BertConfig.from_json_file(os.path.join(args.pretrained_model,"config.json"))
	bert_config.num_hidden_layers = args.depth
	model = model.from_pretrained (args.resume_path, label2id=label2id, device=args.device,config=bert_config)
	model = model.to (args.device)

	trainer = Trainer (n_gpu=args.n_gpu,
	                   model=model,
	                   logger=logger,
	                   tb_logger=None,
	                   optimizer=None,
	                   lr_scheduler=None,
	                   label2id=label2id,
	                   grad_clip=args.grad_clip,
	                   model_checkpoint=None,
	                   gradient_accumulation_steps=args.gradient_accumulation_steps,
					   partial = args.partial,
	                   trigger = args.trigger)
	split = True
	if split:
		diff = ["e", "m", "h"]

		results = {}
		for d in diff:
			test_data = processor.read_type_data (os.path.join (config['data_dir'], "test_gold_all.jsonl"), type=d)
			test_dataset = convert_data_to_tensor (processor=processor,
			                                       args=args,
			                                       data=test_data,
			                                       type=d,
			                                       cache_sample_path=None,
			                                       cache_feature_path=None,
			                                       save_cache=False)
			test_sampler = SequentialSampler (test_dataset)
			test_dataloader = DataLoader (test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
			info, ex_info,class_info = trainer.valid_epoch (test_dataloader)
			results[d] = [class_info,ex_info, info]

		res = json.dumps (results)
		fileObject = open (os.path.join (args.model_path, 'result.json'), 'w')
		fileObject.write (res)
		fileObject.close ()
		prf = ["precision", "recall", "f1"]
		ex_prf = ["ex_p","ex_r","ex_f1"]
		types = ["COM", "INV", "ROU", "AMO"]
		logger.info("Eval results:")
		for d in diff:
			values = []
			class_info,ex_info, info = results[d]
			for t in types:
				cv = class_info[t]
				for k in prf:
					values.append ("{:.4f}".format (cv[k]))
			for k in prf:
				values.append ("{:.4f}".format (info[k]))
			for k in ex_prf:
				values.append("{:.4f}".format (ex_info[k]))
			show_info = f'diff:{d},' + ",".join (values)
			logger.info (show_info)







def main():
	parser = ArgumentParser ()
	parser.add_argument ("--arch", default='bert_crf', type=str)
	parser.add_argument ("--type", default='', type=str)
	parser.add_argument ("--do_train", action='store_true')
	parser.add_argument ("--do_test", action='store_true')
	parser.add_argument ("--do_predict", action='store_true')
	parser.add_argument ("--save_best", action='store_true')
	parser.add_argument ("--do_lower_case", action='store_true')
	parser.add_argument ("--early_stop", action='store_true')
	parser.add_argument ('--data_name', default='datagrand', type=str)
	parser.add_argument ('--optimizer', default='adam', type=str, choices=['adam', 'lookahead'])
	parser.add_argument ('--markup', default='bios', type=str, choices=['bio', 'bios'])
	parser.add_argument ('--checkpoint', default=900000, type=int)
	parser.add_argument ("--epochs", default=30, type=int)
	parser.add_argument ('--fold', default=0, type=int)
	# --resume_path = src/output/checkpoints/bert_lstm_crf_bios_fold_0/checkpoint-epoch-30'
	parser.add_argument ("--tea_resume_path", default='', type=str)
	parser.add_argument("--stu_resume_path", default='', type=str)
	parser.add_argument("--resume_path", default='', type=str)
	parser.add_argument("--depth",default=12,type=int)
	parser.add_argument ("--mode", default='max', type=str)
	parser.add_argument ("--monitor", default='f1', type=str)
	parser.add_argument ("--local_rank", type=int, default=-1)
	parser.add_argument ("--sorted", default=1, type=int, help='1:True  0:False ')
	parser.add_argument ("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
	parser.add_argument ('--gradient_accumulation_steps', type=int, default=2)
	parser.add_argument ("--train_batch_size", default=16, type=int)
	parser.add_argument ('--eval_batch_size', default=64, type=int)
	parser.add_argument ("--train_max_seq_len", default=256, type=int)
	parser.add_argument ("--eval_max_seq_len", default=256, type=int)
	parser.add_argument ('--loss_scale', type=float, default=0)
	parser.add_argument ("--warmup_proportion", default=0.05, type=float)
	parser.add_argument ("--weight_decay", default=0.01, type=float)
	parser.add_argument ("--adam_epsilon", default=1e-8, type=float)
	parser.add_argument ("--grad_clip", default=5.0, type=float)
	parser.add_argument ("--learning_rate", default=1e-4, type=float)
	parser.add_argument("--ex_learning_rate",default=1e-4,type=float)
	parser.add_argument ('--seed', type=int, default=42)
	parser.add_argument ("--no_cuda", action='store_true')
	parser.add_argument ("--partial", action='store_true')
	parser.add_argument ("--trigger", action='store_true')
	parser.add_argument("--test_mode",type=int,default=0)
	parser.add_argument("--alpha", type=float, default=1.0)
	parser.add_argument("--pretrained_model",type=str,default="pretrained_model")
	args = parser.parse_args ()


	args.device = torch.device (f"cuda" if torch.cuda.is_available () and not args.no_cuda else "cpu")

	if args.type:
		args.arch += f"_{args.type}"

	# name_str = "_bs-{}_lr-{}_len-{}".format(args.train_batch_size,args.learning_rate,args.train_max_seq_len)

	args.model_path = config['output'] / args.arch
	args.model_path.mkdir (exist_ok=True)
	# Good practice: save your training arguments together with the trained model
	torch.save (args, args.model_path / 'training_args.bin')
	seed_everything (args.seed)
	init_logger (log_file=args.model_path / f"{args.arch}.log")

	logger.info ("Training/evaluation parameters %s", args)

	if args.do_train:
		run_train (args)

	if args.do_test:
		run_test (args)



if __name__ == '__main__':
	main ()

