import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from kobert_tokenizer import KoBERTTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


device = 'cuda'

class ReviewClassification(nn.Module):
	def __init__(self, n_output=2):
		super(ReviewClassification, self).__init__()
		#self.gpt = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
		#self.transformer = self.gpt.transformer
		self.bert, self.vocab = get_pytorch_kobert_model()
		'''
		self.gate_ship = nn.Linear(768, 512)
		self.gate_quality = nn.Linear(768, 512)
		self.gate_size = nn.Linear(768, 512)
		self.projection = nn.Linear(512*3, n_output)
		'''
		#self.size_proj = nn.Linear(768, 1024)
		#self.projection = nn.Linear(1024, n_output)
		self.projection = nn.Linear(768, n_output)


		for param in self.bert.parameters():
			param.requires_grad = True


	def gen_attention_mask(self, token_ids, valid_length):
		att_mask = torch.zeros_like(token_ids)
		for i, v in enumerate(valid_length):
			att_mask[i][:v] = 1
		return att_mask.float()



    # transformer의 아웃풋을 쓰는게 맞는가 아니면 gpt의 아웃풋을 쓰는게 맞는가
	def forward(self, inputs, valid_length, segment_ids):
		#outputs_tf = outputs_tf[:, -1].contiguous()
		att_mask = self.gen_attention_mask(inputs, valid_length)

		seq_output, pooled_output = self.bert(
				input_ids=inputs,
				attention_mask=att_mask.to(inputs.device),
				token_type_ids=segment_ids.long().to(inputs.device),
				)

		'''
		ship = self.gate_ship(pooled_output)
		quality = self.gate_quality(pooled_output)
		size = self.gate_size(pooled_output)
		concat = torch.cat([ship, quality, size], axis=1)
		'''
		outputs = F.dropout(pooled_output, training=self.training, p=0.2)
		#outputs = self.size_proj(outputs)
		outputs = self.projection(outputs)
		#outputs = self.projection_cls(outputs_tf)
		return F.softmax(outputs, dim=-1)

def load_reviews(path):
	data_path = os.path.join(data_root_path, path)
	label_list = []
	review_list = []
	with open(data_path, 'r') as f:
		lines = f.readlines()
	for line in lines:
		words = line.split(' ')
		text = ' '.join(words[:-1])
		review_list.append(text)
		label_list.append(words[-1])

	return review_list, label_list

def convert2input(review_list):
	token_list = []
	valid_length_list = []
	segment_id_list = []
	for review in review_list:
		sentence = transform([review])
		token_list.append(torch.LongTensor(np.array([sentence[0]])))
		valid_length_list.append(sentence[1])
		segment_id_list.append(torch.LongTensor(np.array([sentence[2]])))

	return token_list, valid_length_list, segment_id_list

def predict(model, tokens, valid_lengths, segment_ids):
	pred_list = []
	for token, valid_length, segment_id in tqdm(zip(tokens, valid_lengths, segment_ids), total=len(tokens), desc='predicting...'):
		token = token.to(device)
		segment_id = segment_id.to(device)
		outputs = model(token, [valid_length], segment_id)
		prob, predict = outputs.max(dim=1)
		pred_list.append(predict.item())
	return np.array(pred_list, dtype=np.int32)



model_root_path = '../models/'
model_path_list = ['ship.pt', 'size.pt', 'quality.pt']
data_root_path = '../../../data/reviews/split_text/'
data_path_list = ['att_split_text0.txt', 'att_split_text1.txt', 'att_split_text2.txt']

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
_, vocab = get_pytorch_kobert_model()

transform = nlp.data.BERTSentenceTransform(
		tokenizer.tokenize,
		max_seq_length=100,
		vocab=vocab,
		pad=True,
		pair=False
		)



for i, (name, data_path) in enumerate(zip(model_path_list, data_path_list)):
	model_path = os.path.join(model_root_path, name)
	review_list, labels = load_reviews(data_path)
	_, review_list, _, labels = train_test_split(review_list, labels, test_size=0.2, shuffle=True, random_state=27)
	tokens, valid_lengths, segment_ids = convert2input(review_list)
	model = ReviewClassification(3)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	model.to(device)
	
	preds = predict(model, tokens, valid_lengths, segment_ids)
	labels = np.array(labels, dtype=preds.dtype)

	#precision, recall, f1, support = precision_recall_fscore_support(labels, preds)
	precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='macro')
	accuracy = accuracy_score(labels, preds)
	'''
	for j in range(3):
		print(f'precision: {precision[j]:.3f}, recall: {recall[j]:.3f}, f1: {f1[j]:.3f}, support: {support[j]}')
	'''
	print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, support: {support}')

	print(f'accuracy: {accuracy:.3f}')

	



