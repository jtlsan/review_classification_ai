from kobert.pytorch_kobert import get_pytorch_kobert_model
import torch.nn as nn
import torch
import torch.nn.functional as F

class ReviewClassification(nn.Module):
	def __init__(self, n_output=2):
		super(ReviewClassification, self).__init__()
		self.bert, self.vocab = get_pytorch_kobert_model() 
		self.projection = nn.Linear(768, n_output)
		
		for param in self.bert.parameters():
			param.requires_grad = True
		
	def gen_attention_mask(self, token_ids, valid_length):
		att_mask = torch.zeros_like(token_ids)
		for i, v in enumerate(valid_length):
			att_mask[i][:v] = 1
		return att_mask.float()
		
	def forward(self, inputs, valid_length, segment_ids):
		att_mask = self.gen_attention_mask(inputs, valid_length)
		seq_output, pooled_output = self.bert(
				input_ids=inputs, 
				attention_mask=att_mask.to(inputs.device), 
				token_type_ids=segment_ids.long(),
				)

		outputs = F.dropout(pooled_output, training=self.training, p=0.2)
		outputs = self.projection(outputs)
		return F.softmax(outputs, dim=-1)
