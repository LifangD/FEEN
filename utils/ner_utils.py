from collections import Counter

import torch


def get_entity_bios(seq, id2label):
	"""Gets entities from sequence.
	note: BIOS
	Args:
		seq (list): sequence of labels.
	Returns:
		list: list of (chunk_type, chunk_start, chunk_end).
	Example:
		>>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
		>>> get_entity_bios(seq)
		[['PER', 0,1], ['LOC', 3, 3]]
	"""
	chunks = []
	chunk = [-1, -1, -1]
	for indx, tag in enumerate (seq):
		if not isinstance (tag, str):
			tag = id2label[tag]
		if tag.startswith ("S-"):
			if chunk[2] != -1:
				chunks.append (chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[2] = indx
			chunk[0] = tag.split ('-')[1]
			chunks.append (chunk)
			chunk = (-1, -1, -1)
		if tag.startswith ("B-"):
			if chunk[2] != -1:
				chunks.append (chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[0] = tag.split ('-')[1]
		elif tag.startswith ('I-') and chunk[1] != -1:
			_type = tag.split ('-')[1]
			if _type == chunk[0]:
				chunk[2] = indx
			if indx == len (seq) - 1:
				chunks.append (chunk)
		else:
			if chunk[2] != -1:
				chunks.append (chunk)
			chunk = [-1, -1, -1]
	return chunks


def get_entity_bio(seq, id2label):
	"""Gets entities from sequence.
	note: BIO
	Args:
		seq (list): sequence of labels.
	Returns:
		list: list of (chunk_type, chunk_start, chunk_end).
	Example:
		>>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
		>>> get_entity_bio(seq)
		[['PER', 0,1], ['LOC', 3, 3]]
	"""
	chunks = []
	chunk = [-1, -1, -1]
	for indx, tag in enumerate (seq):
		if not isinstance (tag, str):
			tag = id2label[tag]

		if tag.startswith ("B-"):
			if chunk[2] != -1:
				chunks.append (chunk)
			chunk = [-1, -1, -1]
			chunk[1] = indx
			chunk[0] = tag.split ('-')[1]
			chunk[2] = indx
			if indx == len (seq) - 1:
				chunks.append (chunk)

		elif tag.startswith ('I-') and chunk[1] != -1:
			_type = tag.split ('-')[1]
			if _type == chunk[0]:
				chunk[2] = indx

			if indx == len (seq) - 1:
				chunks.append (chunk)
		else:
			if chunk[2] != -1:
				chunks.append (chunk)
			chunk = [-1, -1, -1]
	return chunks


def get_entities(seq, id2label, markup='bio'):
	'''
	:param seq:
	:param id2label:
	:param markup:
	:return:
	'''
	assert markup in ['bio', 'bios']
	if markup == 'bio':
		return get_entity_bio (seq, id2label)
	else:
		return get_entity_bios (seq, id2label)


class SeqEntityScore (object):
	def __init__(self, id2label):
		self.id2label = id2label
		self.reset ()

	def reset(self):
		self.origins = []
		self.founds = []
		self.rights = []

	def compute(self, origin, found, right):
		recall = 0 if origin == 0 else (right / origin)
		precision = 0 if found == 0 else (right / found)
		f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
		return recall, precision, f1

	def result(self):
		class_info = {}
		origin_counter = Counter ([x[0] for x in self.origins])
		found_counter = Counter ([x[0] for x in self.founds])
		right_counter = Counter ([x[0] for x in self.rights])
		for type_, count in origin_counter.items ():
			origin = count
			found = found_counter.get (type_, 0)
			right = right_counter.get (type_, 0)
			recall, precision, f1 = self.compute (origin, found, right)
			class_info[type_] = {"precision": round (precision, 4), 'recall': round (recall, 4), 'f1': round (f1, 4)}
		origin = len (self.origins)
		found = len (self.founds)
		right = len (self.rights)
		recall, precision, f1 = self.compute (origin, found, right)
		
		ex_o = len(self.origins)-origin_counter["ROU"]
		ex_f = len(self.founds) - found_counter.get("ROU",0)
		ex_r = len (self.rights) - right_counter.get ("ROU", 0)
		ex_r,ex_p,ex_f1 = self.compute(ex_o,ex_f,ex_r)
		total = {'precision': precision, 'recall': recall, 'f1': f1}
		ex_total = {'ex_p': ex_p, 'ex_r':ex_r, 'ex_f1': ex_f1}
		return total,ex_total,class_info

	def update(self, label_paths, pred_paths):
		'''
		labels_paths: [[],[],[],....]
		pred_paths: [[],[],[],.....]

		:param label_paths:
		:param pred_paths:
		:return:
		Example:
			>>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
			>>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
		'''
		r = {"founds":[],"origins":[],"rights":[]}
		for label_path, pre_path in zip (label_paths, pred_paths):
			label_entities = get_entities (label_path, self.id2label)
			pre_entities = get_entities (pre_path, self.id2label)
			self.origins.extend (label_entities)
			self.founds.extend (pre_entities)
			self.rights.extend ([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
			
			r["founds"].extend(pre_entities)
			r["origins"].extend(label_entities)
			r["rights"].extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
		return r


class SpanEntityScore (object):
	def __init__(self, id2label):
		self.id2label = id2label
		self.reset ()

	def reset(self):
		self.origins = []
		self.founds = []
		self.rights = []

	def compute(self, origin, found, right):
		recall = 0 if origin == 0 else (right / origin)
		precision = 0 if found == 0 else (right / found)
		f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
		return recall, precision, f1

	def result(self):
		class_info = {}
		origin_counter = Counter ([self.id2label[x[0]] for x in self.origins])
		found_counter = Counter ([self.id2label[x[0]] for x in self.founds])
		right_counter = Counter ([self.id2label[x[0]] for x in self.rights])
		for type_, count in origin_counter.items ():
			origin = count
			found = found_counter.get (type_, 0)
			right = right_counter.get (type_, 0)
			recall, precision, f1 = self.compute (origin, found, right)
			class_info[type_] = {"acc": round (precision, 4), 'recall': round (recall, 4), 'f1': round (f1, 4)}
		origin = len (self.origins)
		found = len (self.founds)
		right = len (self.rights)
		recall, precision, f1 = self.compute (origin, found, right)
		return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

	def update(self, true_subject, pred_subject):
		self.origins.extend (true_subject)
		self.founds.extend (pred_subject)
		self.rights.extend ([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


def bert_extract_item(start_logits, end_logits):
	S = []
	start_pred = torch.argmax (start_logits, -1).cpu ().numpy ()[0][1:-1]
	end_pred = torch.argmax (end_logits, -1).cpu ().numpy ()[0][1:-1]
	for i, s_l in enumerate (start_pred):
		if s_l == 0:
			continue
		for j, e_l in enumerate (end_pred[i:]):
			if s_l == e_l:
				S.append ((s_l, i, i + j))
				break
	return S


if __name__ == "__main__":
	seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
	print (get_entities (seq, None))
