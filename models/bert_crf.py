from modules.crf import *
from modules.normalization import LayerNorm
from modules.bert.modeling_bert import BertModel,BertEmbeddings
from modules.bert.modeling_bert import BertPreTrainedModel


class BERTCRF (BertPreTrainedModel):
	def __init__(self, config, label2id, device,):
		super (BERTCRF, self).__init__ (config)
		self.bert = BertModel (config)
		self.dropout = nn.Dropout (config.hidden_dropout_prob)

		self.init_weights ()
		self.layer_norm = LayerNorm (config.hidden_size)
		self.classifier = nn.Linear (config.hidden_size, len (label2id))
		self.crf = CRF (tagset_size=len (label2id), tag_dictionary=label2id, device=device,
		                is_bert=True)  # the original

	def forward_f(self, input_ids, token_type_ids=None, attention_mask=None,trigger_mask=None):
		outputs = self.bert (input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask,trigger_mask=trigger_mask)
		sequence_output = outputs[0]
		sequence_output = self.dropout (sequence_output)
		sequence_output = self.layer_norm (sequence_output)
		features = self.classifier (sequence_output)
		return features

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, trigger_mask=None,labels=None, input_lens=None,
	            one_hot_labels=None):
		features = self.forward_f (input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,trigger_mask=trigger_mask)
		if one_hot_labels is not None:
			loss = self.crf.partial_loss (features, input_lens, one_hot_labels, )
			return features, loss
		else:
			loss = self.crf.calculate_loss (features, labels, input_lens)
			return features,loss

	def unfreeze(self, start_layer=6, end_layer=12):
		def children(m):
			return m if isinstance (m, (list, tuple)) else list (m.children ())

		def set_trainable_attr(m, b):
			m.trainable = b
			for p in m.parameters ():
				p.requires_grad = b

		def apply_leaf(m, f):
			c = children (m)
			if isinstance (m, nn.Module):
				f (m)
			if len (c) > 0:
				for l in c:
					apply_leaf (l, f)

		def set_trainable(l, b):
			apply_leaf (l, lambda m: set_trainable_attr (m, b))

		# You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
		set_trainable (self.bert, False)
		for i in range (start_layer, end_layer):
			set_trainable (self.bert.encoder.layer[i], True)


class CNNLSTMCRF(nn.Module):
	'''
	Neural Chinese Named Entity Recognition via CNN-LSTM-CRF and Joint Training with Word Segmentation
	total CNN_filter_number=400, lstm_size=200 kernel_size/window_size ranges from [2,5]
	word2vec embedding
	dropout0.2

	'''
	def __init__(self, config, label2id, device,):
		super (CNNLSTMCRF, self).__init__ ()
		filter_sizes = [3,5]
		self.BERTEm = BertEmbeddings(config)
		self.CNN = nn.ModuleList([nn.Conv1d(in_channels=config.hidden_size,
											out_channels=100,
											kernel_size=fsz,
											padding=int((fsz-1)/2))
								  			for fsz in filter_sizes],
								 )
		self.BiLSTM= nn.LSTM(input_size=200,
							  hidden_size=200,
							  batch_first=True,
							  num_layers=1,
							  dropout=0,
							  bidirectional=True)

		self.dropout = nn.Dropout(0.2)
		self.classifier = nn.Linear(400, len(label2id))
		self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device,
					   is_bert=True)  # the original

	def forward_f(self, input_ids, token_type_ids=None):
		embeddings = self.BERTEm (input_ids, token_type_ids=token_type_ids).transpose(1,2)

		x_conv = [conv(embeddings).transpose(1,2) for conv in self.CNN]
		x_conv = torch.cat(x_conv,-1)
		sequence_output,_ = self.BiLSTM (x_conv)
		sequence_output = self.dropout(sequence_output)

		features = self.classifier (sequence_output)
		return features

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, trigger_mask=None,labels=None, input_lens=None,
	            one_hot_labels=None):
		features = self.forward_f (input_ids, token_type_ids=token_type_ids,)
		if one_hot_labels is not None:
			loss = self.crf.partial_loss (features, input_lens, one_hot_labels, )
			return features, loss
		else:
			loss = self.crf.calculate_loss (features, labels, input_lens)
			return features,loss