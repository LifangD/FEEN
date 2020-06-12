import torch
from torch.utils.data import TensorDataset

from utils import ProgressBar
from utils.tools import load_pickle
from utils.tools import logger
from modules.bert.tokenization_bert import BertTokenizer
import jsonlines


ANCHORS =  {'Pre-A轮', 'C轮', 'IPO上市', 'B轮', 'E轮', 'Pre-B轮', 'D+轮', 'F轮-上市前', '新三板定增', '新三板', 'D轮', 'B+轮', 'C+轮', 'A+轮', 'A轮', '天使轮', '战略投资', '种子轮', 'IPO上市后'}
def lower_string(s):
	res = ""
	for c in s:
		res+=c.lower()
	return res

class InputExample (object):
	def __init__(self, guid, text_a, text_b=None, label=None, match=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label
		self.match = match


class InputFeature (object):
	'''
	A single set of features of data.
	'''

	def __init__(self, input_ids, input_mask, trigger_mask,segment_ids, label_id, one_hot_labels, input_len):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.trigger_mask = trigger_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.one_hot_labels = one_hot_labels
		self.input_len = input_len


class CustomTokenizer (BertTokenizer):
	def __init__(self, vocab_file, min_freq_words=None, do_lower_case=False):
		super ().__init__ (vocab_file=str (vocab_file), do_lower_case=do_lower_case)
		self.vocab_file = str (vocab_file)
		self.do_lower_case = do_lower_case
		self.min_freq_words = min_freq_words

	def tokenize(self, text):
		_tokens = []
		for c in text:
			if self.do_lower_case:
				c = c.lower ()
			if c in self.vocab:
				if self.min_freq_words is not None:
					if c in self.min_freq_words:
						continue
				_tokens.append (c)
			else:
				_tokens.append ('[UNK]')
		return _tokens


class BertProcessor (object):
	"""Base class for data converters for sequence classification data sets."""

	def __init__(self, vocab_path, do_lower_case, test_mode,min_freq_words=None,):
		# self.tokenizer = BertTokenizer(vocab_path,do_lower_case)
		self.tokenizer = CustomTokenizer (vocab_path, min_freq_words, do_lower_case, )
		self.test_mode=test_mode # 0,4

	def get_train(self, data_file):
		"""Gets a collection of `InputExample`s for the train set."""
		return self.read_data (data_file)

	def get_dev(self, data_file):
		"""Gets a collection of `InputExample`s for the dev set."""
		return self.read_data (data_file)

	def get_test(self, lines):
		return lines

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		return ["O", "B-COM", "I-COM", "B-INV", "I-INV", "B-ROU", "I-ROU", "B-AMO", "I-AMO", "[CLS]", "[SEP]"]

	@classmethod
	def read_data(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		if 'pkl' in str (input_file):
			lines = load_pickle (input_file)
		if "jsonl" in str (input_file):
			lines = []
			with jsonlines.open (input_file) as reader:
				for row in reader:
					lines.append (row)

		else:
			lines = input_file
		return lines

	def read_type_data(cls, input_file, type):

		with jsonlines.open (input_file) as reader:
			lines = []
			for line in reader:
				e_d = line["guid"].split ("_")[1]
				# e_m = sum(line["cira_match"])
				# datasets["%s_%d"%(e_d,e_m)].append(line)
				if e_d in type:
					lines.append (line)
		logger.info ("type {} number = {}".format (type, len (lines)))
		return lines

	def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
		# This is a simple heuristic which will always truncate the longer sequence
		# one token at a time. This makes more sense than truncating an equal percent
		# of tokens from each, since if one sequence is very short then each token
		# that's truncated likely contains more information than a longer sequence.
		while True:
			total_length = len (tokens_a) + len (tokens_b)
			if total_length <= max_length:
				break
			if len (tokens_a) > len (tokens_b):
				tokens_a.pop ()
			else:
				tokens_b.pop ()

	def create_examples(self, lines, example_type, cached_file, save_cache):
		'''
		Creates examples for data
		'''
		label_list = self.get_labels ()
		if cached_file and cached_file.exists ():
			logger.info ("Loading examples from cached file %s", cached_file)
			examples = torch.load (cached_file)
		else:
			pbar = ProgressBar (n_total=len (lines), desc='create examples')
			examples = []
			for i, line in enumerate (lines):
				#if i>20:break # for quik debug
				guid = '%s-%d' % (example_type, i)
				label = line['tags']
				text_a = line['info']
				text_b = None
				match = line["cira_match"]
			
				if self.test_mode==4 and sum(match)<4:
					continue
				else:
					examples.append (InputExample (guid=guid, text_a=text_a, text_b=text_b, label=label, match=match))
				pbar (step=i)

			if save_cache:
				logger.info ("Saving examples into cached file %s", cached_file)

				torch.save (examples, cached_file)
		return examples

	def create_features(self, examples, max_seq_len, cached_file, save_cache=False):
		'''
		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids:   0   0   0   0  0     0   0
		'''
		max_lenth = 0
		if cached_file and cached_file.exists ():
			logger.info ("Loading features from cached file %s", cached_file)
			features = torch.load (cached_file)
		else:
			label_list = self.get_labels ()
			label2id = {label: i for i, label in enumerate (label_list, 0)}

			pbar = ProgressBar (n_total=len (examples), desc='create features')
			features = []

			for ex_id, example in enumerate (examples):

				# textlist = []
				# for sentence in example.text_a:
				#     textlist.extend(list(sentence))
				textlist = list (example.text_a)
				if len (textlist) > max_lenth:
					max_lenth = len (textlist)

				tokens = self.tokenizer.tokenize (textlist)
				labels = example.label
				match = example.match
				if len (tokens) >= max_seq_len - 2:
					tokens = tokens[0:(max_seq_len - 2)]
					labels = labels[0:(max_seq_len - 2)]
				ntokens = []
				segment_ids = []
				label_ids = []
				one_hot_labels = []
				ntokens.append ("[CLS]")
				segment_ids.append (0)
				mask_tags = [1] * len (label2id)
				mask_tags[label2id["[CLS]"]] = 0
				one_hot_labels.append (mask_tags)
				label_ids.append (label2id["[CLS]"])

				possible_tags = [1] * len (label2id)
				trigger_mask = []
				trigger_mask.append(0)
				if sum (match) < 4:
					possible_tags[0] = 0
				
					#assert  match[0]==1
					# if match[0] < 1:
					# 	possible_tags[1:3] = [0,0]
					if match[1] < 1:
						possible_tags[3:5] = [0,0]
					#assert match[2]==1
					# if match[2] < 1:
					# 	possible_tags[5:7] = [0, 0]
					if match[3] < 1:
						possible_tags[7:9] = [0, 0]
					


				for i, token in enumerate (tokens):
					ntokens.append (token)
					segment_ids.append (0)
					label_ids.append (label2id[labels[i]])
					if "ROU" in labels[i]:
						trigger_mask.append(1)
					else:
						trigger_mask.append(0)

					if sum (match) < 4 and labels[i] == 'O' and (token not in range(7993,8029)) and (token not in range(8039,8051)):
						one_hot_labels.append (possible_tags)

					else:
						mask_tags = [1] * len (label2id)
						mask_tags[label2id[labels[i]]] = 0
						one_hot_labels.append (mask_tags)

				ntokens.append ("[SEP]")
				segment_ids.append (0)
				label_ids.append (label2id["[SEP]"])
				mask_tags = [1] * len (label2id)
				mask_tags[label2id["[SEP]"]] = 0
				one_hot_labels.append (mask_tags)
				trigger_mask.append(0)

				input_ids = self.tokenizer.convert_tokens_to_ids (ntokens)
				input_mask = [1] * len (input_ids)
				
				input_len = len (label_ids)

				while len (input_ids) < max_seq_len:
					input_ids.append (0)
					input_mask.append (0)
					segment_ids.append (0)
					label_ids.append (0)
					one_hot_labels.append ([1] * len (label2id))
					trigger_mask.append(0)

				assert len (input_ids) == max_seq_len
				assert len (input_mask) == max_seq_len
				assert len (segment_ids) == max_seq_len
				assert len (label_ids) == max_seq_len
				assert  len(one_hot_labels) == max_seq_len
				assert len(one_hot_labels) == max_seq_len

				for i in range (len (one_hot_labels)):
					if len(one_hot_labels[i])<11:
						logger.info (
							"one-hot labels: pos:%d, %s" % (i, " ".join ([str (x) for x in one_hot_labels[i]])))
					# if ex_id < 2:
						logger.info ("*** Example ***")
						logger.info ("guid: %s" % (example.guid))
						logger.info ("tokens: %s" % " ".join ([str (x) for x in tokens]))
						logger.info ("input_ids: %s" % " ".join ([str (x) for x in input_ids]))
						logger.info ("input_mask: %s" % " ".join ([str (x) for x in input_mask]))
						logger.info ("segment_ids: %s" % " ".join ([str (x) for x in segment_ids]))
						logger.info (
							"label: %s id: %s" % (" ".join (example.label), " ".join ([str (x) for x in label_ids])))

				features.append (
					InputFeature (input_ids=input_ids,
					              input_mask=input_mask,
					              trigger_mask =trigger_mask,
					              segment_ids=segment_ids,
					              label_id=label_ids,
					              one_hot_labels=one_hot_labels,
					              input_len=input_len))

				pbar (step=ex_id)
			if save_cache:
				logger.info ("Saving features into cached file %s", cached_file)
				torch.save (features, cached_file)
		logger.info ("max_seq_lenth = {}".format (max_lenth))
		return features


	def create_dataset(self, features, is_sorted=False):
		# Convert to Tensors and build dataset
		if is_sorted:
			logger.info ("sorted data by th length of input")
			features = sorted(features, key=lambda x: x.input_len, reverse=True)
		all_input_ids = torch.tensor ([f.input_ids for f in features], dtype=torch.long)
		all_input_mask = torch.tensor ([f.input_mask for f in features], dtype=torch.long)
		all_trigger_mask = torch.tensor ([f.trigger_mask for f in features], dtype=torch.long)
		all_segment_ids = torch.tensor ([f.segment_ids for f in features], dtype=torch.long)
		all_label_ids = torch.tensor ([f.label_id for f in features], dtype=torch.long)
		all_input_lens = torch.tensor ([f.input_len for f in features], dtype=torch.long)
		all_one_hot_labels = torch.tensor ([f.one_hot_labels for f in features], dtype=torch.long)
		dataset = TensorDataset (all_input_ids, all_input_mask, all_trigger_mask,all_segment_ids, all_label_ids, all_input_lens,
		                         all_one_hot_labels)
	
		return dataset

	def get_anchor_pos(self,sentence):
		sentence = lower_string(sentence)
		anchor_values = []

		for anchor_value in ANCHORS:
			anchor_value = lower_string(anchor_value)
			b_idx = sentence.find(anchor_value)

			if b_idx != -1:
				e_idx = b_idx + len(list(anchor_value))
				anchor_values.append([b_idx,e_idx])

		return anchor_values

	def create_test_features(self, sentence, max_seq_len):
		'''
		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids:   0   0   0   0  0     0   0
		'''

		label_list = self.get_labels()
		label2id = {label: i for i, label in enumerate(label_list, 0)}
		textlist = list(sentence)
		input_len = len(textlist)
		tokens = self.tokenizer.tokenize(textlist)

		if len(tokens) >= max_seq_len - 2:
			tokens = tokens[0:(max_seq_len - 2)]

		ntokens = []
		segment_ids = []

		one_hot_labels = []
		ntokens.append("[CLS]")
		segment_ids.append(0)
		mask_tags = [1] * len(label2id)
		mask_tags[label2id["[CLS]"]] = 0
		one_hot_labels.append(mask_tags)





		for i, token in enumerate(tokens):
			ntokens.append(token)
			segment_ids.append(0)

		ntokens.append("[SEP]")
		segment_ids.append(0)
		mask_tags = [1] * len(label2id)
		mask_tags[label2id["[SEP]"]] = 0

		input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
		input_mask = [1] * len(input_ids)


		while len(input_ids) < max_seq_len:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)


		assert len(input_ids) == max_seq_len
		assert len(input_mask) == max_seq_len
		assert len(segment_ids) == max_seq_len


		anchor_poses = self.get_anchor_pos(sentence)
		trigger_masks = []

		trigger_words=[]
		if anchor_poses != []:
			for anchor_pos in anchor_poses:
				trigger_mask = [0] * max_seq_len
				for i in range(anchor_pos[0], anchor_pos[1]):
					trigger_mask[i+1] = 1 # 前面有CLS
				trigger_masks.append(trigger_mask)
				trigger_words.append(sentence[anchor_pos[0]:anchor_pos[1]])
		else:
			trigger_masks=[input_mask]
		return input_ids,segment_ids,input_mask,trigger_masks,input_len,trigger_words




def convert_data_to_tensor(processor, args, data, type, cache_sample_path, cache_feature_path, save_cache):
	examples = processor.create_examples (lines=data, example_type=type, cached_file=cache_sample_path,
	                                      save_cache=save_cache)
	features = processor.create_features (examples=examples, max_seq_len=args.train_max_seq_len,
	                                      cached_file=cache_feature_path, save_cache=save_cache)
	dataset = processor.create_dataset (features, is_sorted=args.sorted,)
	return dataset