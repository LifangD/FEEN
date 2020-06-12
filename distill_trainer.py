import os

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from utils.ner_utils import SeqEntityScore
from utils.progressbar import ProgressBar
from utils.tools import AverageMeter
from utils.tools import prepare_device
from utils.tools import seed_everything



class DistillTrainer (object):
	def __init__(self, tmodel,smodel, n_gpu, logger, tb_logger, optimizer, lr_scheduler,
	             label2id, gradient_accumulation_steps, grad_clip=0.0, early_stopping=None,
	             model_checkpoint=None, partial=False,trigger=False,alpha=1):

		self.n_gpu = n_gpu
		self.tmodel = tmodel
		self.smodel = smodel
		self.logger = logger
		self.tb_logger = tb_logger
		self.optimizer = optimizer
		self.label2id = label2id
		self.grad_clip = grad_clip
		self.lr_scheduler = lr_scheduler
		self.early_stopping = early_stopping
		self.model_checkpoint = model_checkpoint
		self.gradient_accumulation_steps = gradient_accumulation_steps
		self.device, _ = prepare_device (n_gpu)
		self.id2label = {y: x for x, y in label2id.items ()}
		self.entity_score = SeqEntityScore (self.id2label)
		self.start_epoch = 0
		self.global_step = 0
		self.partial = partial
		self.trigger = trigger
		self.alpha=alpha


	def save_info(self, epoch, best):
		model_save = self.smodel.module if hasattr (self.smodel, 'module') else self.smodel
		state = {"model": model_save,
		         'epoch': epoch,
		         'best': best,
		         'optimizer': self.optimizer.state_dict ()}
		return state

	def valid_epoch(self, data_loader):
		pbar = ProgressBar (n_total=len (data_loader), desc='Evaluating')
		self.entity_score.reset ()
		valid_loss = AverageMeter ()
		#output_file = jsonlines.open("data/case_out.jsonl","w")
		for step, batch in enumerate (data_loader):

			batch = tuple (t.to (self.device) for t in batch)
			
			input_ids, input_mask, trigger_mask,segment_ids, label_ids, input_lens, one_hot_labels = batch
			
			if not self.trigger:
				trigger_mask=None
			if not self.partial:
				one_hot_labels=None
			input_lens = input_lens.cpu ().detach ().numpy ().tolist ()
			self.smodel.eval ()
			with torch.no_grad ():
				features, loss = self.smodel (input_ids, segment_ids, input_mask,trigger_mask, label_ids, input_lens,
				                             one_hot_labels=one_hot_labels)
				tags,_= self.smodel.crf._obtain_labels (features,self.id2label,input_lens)
			valid_loss.update (val=loss.item (), n=input_ids.size (0))
			pbar (step=step, info={"loss": loss.item ()})
			label_ids = label_ids.to ('cpu').numpy ().tolist ()

			for i, label in enumerate (label_ids):
				temp_1 = []
				temp_2 = []
				for j, m in enumerate (label):
					if j == 0:
						continue
					elif j==input_lens[i]-1:
						r = self.entity_score.update (pred_paths=[temp_2], label_paths=[temp_1])
						r["input_ids"] = input_ids[i,:].to("cpu").numpy().tolist()
						#output_file.write(r)
						break
					else:
						temp_1.append (self.id2label[label_ids[i][j]])
						temp_2.append (tags[i][j])
		
		valid_info, ex_valid_info,class_info = self.entity_score.result ()
		ex_info = {f'{key}': value for key, value in ex_valid_info.items ()}
		info = {f'{key}': value for key, value in valid_info.items ()}
		info['valid_loss'] = valid_loss.avg
		if 'cuda' in str (self.device):
			torch.cuda.empty_cache ()
		return info, ex_info,class_info

	def train_epoch(self, data_loader):

		pbar = ProgressBar (n_total=len (data_loader), desc='Training')
		tr_loss = AverageMeter ()
		self.tmodel.eval()
		for step, batch in enumerate (data_loader):
			self.smodel.train ()

			batch = tuple (t.to (self.device) for t in batch)

			input_ids, input_mask,trigger_mask, segment_ids, label_ids, input_lens, one_hot_labels = batch
			if not self.partial:
				one_hot_labels = None
			if not self.trigger:
				trigger_mask = None
			lens = input_lens.cpu ().detach ().numpy ().tolist ()
			with torch.no_grad():
				t_features, t_loss,= self.tmodel(input_ids, segment_ids, input_mask,trigger_mask, label_ids, lens, one_hot_labels)
			# 	_,tag_ids = self.tmodel.crf._obtain_labels(t_features, self.id2label, input_lens)
			#
			#
			#
			# teacher_tags = []
			# max_seq_length=256
			# for tag in tag_ids:
			# 	tag+=[0]*(max_seq_length-len(tag))
			# 	teacher_tags.append(tag)
			# teacher_labels = torch.tensor([tag for tag in teacher_tags], dtype=torch.long).cuda()
			# s_features = self.smodel.forward_f(input_ids, segment_ids, input_mask, trigger_mask,)
			# combine_labels = torch.where(label_ids==0,teacher_labels,label_ids)
			#
			# loss = self.smodel.crf.calculate_loss(s_features, combine_labels, input_lens)

			'''
			s_partial_loss = self.smodel.crf.partial_loss(s_features, input_lens, one_hot_labels) # outer signal
			st_loss = self.smodel.crf.calculate_loss(s_features, teacher_labels, input_lens) # sequence-level teacher signal
			kld_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(s_features,-1),F.softmax(t_features,-1)) #word-levle teacher signal
			bs = kld_loss.size(0)
			pad_kld_loss = kld_loss*(input_mask.float().unsqueeze(-1))
			reduced_kld_loss = torch.mean(torch.sum(pad_kld_loss.view(bs,-1),-1)/input_lens.float(),0)
			loss=(1-self.alpha)*s_partial_loss+self.alpha*(reduced_kld_loss+st_loss)
			'''
			s_features = self.smodel.forward_f(input_ids, segment_ids, input_mask, trigger_mask, )
			kld_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(s_features, -1),
													  F.softmax(t_features, -1))  # word-levle teacher signal
			bs = kld_loss.size(0)
			pad_kld_loss = kld_loss * (input_mask.float().unsqueeze(-1))
			loss= torch.mean(torch.sum(pad_kld_loss.view(bs, -1), -1) / input_lens.float(), 0)

			if len (self.n_gpu.split (",")) >= 2:
				loss = loss.mean ()
			if self.gradient_accumulation_steps > 1:
				loss = loss / self.gradient_accumulation_steps

			loss.backward ()
			clip_grad_norm_ (self.smodel.parameters (), self.grad_clip)
			if (step + 1) % self.gradient_accumulation_steps == 0:
				self.optimizer.step ()
				self.optimizer.zero_grad ()
				self.global_step += 1
			tr_loss.update (loss.item (), n=1)
			# self.tb_logger.scalar_summary ("s_partial_loss", s_partial_loss.item(), self.global_step)
			# self.tb_logger.scalar_summary('st_loss',st_loss.item(),self.global_step)
			# self.tb_logger.scalar_summary('kld_loss',reduced_kld_loss.item(),self.global_step)
			pbar (step=step, info={'loss': loss.item ()})
			# if step%5==0:
			# 	self.logger.info("step:{},loss={:.4f}".format(self.global_step,loss.item()))


		info = {'loss': tr_loss.avg}
		if "cuda" in str (self.device):
			torch.cuda.empty_cache ()
		return info

	def train(self, train_data, valid_data, epochs, seed):
		seed_everything (seed)
		for epoch in range (self.start_epoch, int (epochs)):
			self.logger.info (f"Epoch {epoch}/{int(epochs)}")
			train_log = self.train_epoch (train_data)  # {'loss':}
			valid_log, ex_valid_log,class_info = self.valid_epoch (valid_data)
			# valid_log: valid_loss,precision,recall,f1
			# class_info: {"INV":{"recall":}}

			logs = dict (train_log, **valid_log,**ex_valid_log)
			show_info = f'Epoch: {epoch} - ' + "-".join ([f' {key}: {value:.4f} ' for key, value in logs.items ()])
			self.logger.info (show_info)
			self.logger.info ("The entity scores of valid data : ")
			for key, value in class_info.items ():
				info = f'Entity: {key} - ' + "-".join ([f' {key_}: {value_:.4f} ' for key_, value_ in value.items ()])
				self.logger.info (info)

			if self.model_checkpoint:
				state = self.save_info (epoch, best=logs[self.model_checkpoint.monitor])
				self.model_checkpoint.bert_epoch_step (current=logs[self.model_checkpoint.monitor], state=state)

			if hasattr (self.lr_scheduler, 'epoch_step'):
				self.lr_scheduler.epoch_step (metrics=logs[self.model_checkpoint.monitor],
				                              epoch=epoch)  # the monitor is "loss"

			if self.early_stopping:
				self.early_stopping.epoch_step (current=logs[self.early_stopping.monitor])
				if self.early_stopping.stop_training and self.lr_scheduler.lr<1e-6:
					break
