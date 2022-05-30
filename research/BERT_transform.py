'''
main(infile.txt, data_path(dir), model_output_path(.pt), class_cnt)
'''
import numpy as np
import math
import json
import os
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from hanspell import spell_checker
from konlpy.tag import Okt
import re
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert_tokenizer import KoBERTTokenizer
import gluonnlp as nlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support





BOS = '</s>'
EOS = '</s>'
PAD = '<pad>'
MASK = '<unused0>'
stopword_pos = ['Conjunction', 'Exclamation', 'Foreign', 'Josa', 'KoreanParticle', 'Number', 'Punctuation', 'Alpha']
data_cls_cnt = [3, 8, 4, 6, 61, 23, 3, 32, 9, 48, 702, 249, 313, 5104, 2832, 117, 1632, 956, 1, 12, 3, 4, 107, 62, 2, 36, 16]

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
				token_type_ids=segment_ids.long(),
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


class ReviewDataset(Dataset):
	def __init__(self, tokenizer, data, labels, name: str, data_path, save, tok, vocab, max_len=100):
		self.tokenizer = tokenizer
		self.labels = []
		self.sentences = []
		self.max_len = max_len
		self.okt = Okt()
		self.name_sentences = name + '_sentences'
		self.name_labels = name + '_labels'
		self.name_att_masks = name + '_att_masks'
		self.name_token_type_ids = name + '_token_type_ids'
		transform = nlp.data.BERTSentenceTransform(
				tok, max_seq_length=max_len, vocab=vocab, pad=True, pair=False)
		
		label_cnt = [0, 0, 0]
		pbar = tqdm(enumerate(data), total=len(data), desc='setting Datasets...')
		for i, tokens in pbar:
			label = np.int32(labels[i])
			text = ' '.join(tokens)
			sentence = transform([text])
			
			label_cnt[label] += 1
			if label_cnt[label] > 3000:
				continue
			
			self.sentences.append(sentence)
			self.labels.append(label)
		'''
		if save:
			for i, tokens in pbar:
				label = labels[i]
				text = ' '.join(tokens)
				sentence = transform([text])
				
				self.sentences.append(sentence)
				self.labels.append(np.int32(label))
			self._save_dataset(data_path)
		else:
			self._load_dataset(data_path)
		'''


	def __len__(self):
		assert len(self.labels) == len(self.sentences)
		return len(self.labels)

	def __getitem__(self, item):
		return (self.labels[item],
				self.sentences[item],
				)
	
	def _preprocess_text(self, text):
		refined = re.sub('[^가-힣]', '', text)
		refined = refined.replace(' ', '')
		refined = spell_checker.check(refined).checked
		for word, pos in self.okt.pos(refined):
			if pos in stopword_pos:
				refined = refined.replace(word, '')
		
		return refined

	def _gen_path(self, path):
		s_path = os.path.join(path, self.name_sentences + '.npy')
		l_path = os.path.join(path, self.name_labels + '.npy')
		return s_path, l_path

	def _save_dataset(self, path):
		sentences = np.array(self.sentences)
		labels = np.array(self.labels)
		s_path, l_path = self._gen_path(path)

		np.save(s_path, sentences)
		np.save(l_path, labels)

	def _load_dataset(self, path):
		s_path, l_path = self._gen_path(path)
		self.sentences = np.load(s_path).tolist()
		self.labels = np.load(l_path).tolist()


def review_collate_fn(inputs):
	# inputs : (batch_size, (labels, sentences))
	labels = [item[0] for item in inputs]
	data = [item[1][0] for item in inputs]
	valid_length = [item[1][1] for item in inputs]
	segment_ids = [item[1][2] for item in inputs]

	torch.LongTensor(labels)
	torch.LongTensor(data)
	torch.LongTensor(segment_ids)

	return torch.LongTensor(labels), torch.LongTensor(data), valid_length, torch.LongTensor(segment_ids)

def train(model, epoch, criterion, optimizer, train_loader):
	losses = []
	model.train()

	with tqdm(total=len(train_loader), desc=f'Train({epoch})') as pbar:
		for i, value, in enumerate(train_loader):
			labels, inputs, valid_length, segment_ids = value
			labels = labels.to(device)
			inputs = inputs.to(device)
			segment_ids = segment_ids.to(device)

			optimizer.zero_grad()
			outputs = model(inputs, valid_length, segment_ids)
			# output.logits : (batch_size, max_len, vocab_size=51200)
			#output = output.logits

			loss = criterion(outputs, labels)
			loss_val = loss.item()
			losses.append(loss_val)

			loss.backward()
			optimizer.step()
			
			pbar.update(1)
			pbar.set_postfix_str(f'Loss: {loss_val:.3f} ({np.mean(losses):.3f})')

	return np.mean(losses)
	
def eval(model, data_loader, criterion, lr_scheduler, best_f1_mean, model_path, class_cnt):
	matches = []
	model.eval()
	y_true = []
	y_pred = []

	n_word_total = 0
	n_correct_total = 0
	with tqdm(total=len(data_loader), desc=f'Valid') as pbar:
		for i, value in enumerate(data_loader):
			labels, inputs, valid_length, segment_ids = value
			labels = labels.to(device)
			inputs = inputs.to(device)
			segment_ids = segment_ids.to(device)

			outputs = model(inputs, valid_length, segment_ids)
			_, indices = outputs.max(dim=1)

			loss = criterion(outputs, labels)
			loss_val = loss.item()

			match = torch.eq(labels, indices).detach()
			matches.extend(match.cpu())
			y_true.append(labels[0].detach().cpu().numpy())
			y_pred.append(indices[0].detach().cpu().numpy())
			accuracy = np.sum(matches) / len(matches) if 0 < len(matches) else 0

			pbar.update(1)
			pbar.set_postfix_str(f'Acc: {accuracy:.3f}')
		lr_scheduler.step(accuracy)
	

	#labels=[i for i in range(class_cnt)]
	precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
	for i in range(class_cnt):
		print(f'#{i} => precision: {precision[i]:.3f}, recall: {recall[i]:.3f}, f1: {f1[i]:.3f}, support: {support[i]}')

	f1_mean = f1.mean()
	if f1_mean > best_f1_mean:
		print(f'saving model : new_f1_mean = {f1_mean}')
		torch.save(model.state_dict(), model_path)

	
	return accuracy, f1_mean 
				

def main(infile, data_path, model_output_path, class_cnt, save):
	datas = []
	labels = []
	with open(infile, 'r') as f:
		for line in f.readlines():
			tokens = line.split()
			datas.append(tokens[:-1])
			labels.append(tokens[-1].strip())
	X_train, X_valid, y_train, y_valid = train_test_split(datas, labels, test_size=0.2, shuffle=True, random_state=27)
	'''
	tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
			bos_token=BOS,
			eos_token=EOS,
			unk_token='<unk>',
			pad_token=PAD,
			mask_token=MASK)
	'''
	tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
	model = ReviewClassification(class_cnt)

	train_dataset = ReviewDataset(tokenizer, X_train, y_train, tok=tokenizer.tokenize, vocab=model.vocab, name='train', data_path=data_path, save=save)
	valid_dataset = ReviewDataset(tokenizer, X_valid, y_valid, tok=tokenizer.tokenize, vocab=model.vocab, name='valid', data_path=data_path, save=save)
	#train_dataset = ReviewDataset(tokenizer, infile, start_num=0, max_num=10, name='train', data_path=data_path, save=save)
	#valid_dataset = ReviewDataset(tokenizer, infile, start_num=10, max_num=20, name='valid', data_path=data_path, save=save)
	train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=review_collate_fn)
	#num_workers
	valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=review_collate_fn)

	weights = [5, 0.1, 2]
	class_weights = torch.FloatTensor(weights).to(device)
	criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
	#criterion = torch.nn.CrossEntropyLoss()
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_param_group = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.}
			]
	#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	optimizer = torch.optim.Adam(optimizer_param_group, lr=1e-5)
	lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, cooldown=2, verbose=1)
	epochs = 20

	model.to(device)

	best_f1_mean = 0
	for epoch in range(epochs):
		loss = train(model, epoch, criterion, optimizer, train_dataloader)
		accuracy, f1_mean = eval(model, valid_dataloader, criterion, lr_scheduler, best_f1_mean, model_output_path, class_cnt)
		if f1_mean > best_f1_mean:
			best_f1_mean = f1_mean

	torch.save(model.state_dict(), model_output_path)

if __name__ == '__main__':
	data_path = sys.argv[2]
	if os.path.isdir(data_path):
		if not os.listdir(data_path):
			save=True
		else:
			save=False
	else:
		raise RuntimeError('no valid data_path')
	main(sys.argv[1], data_path, sys.argv[3], int(sys.argv[4]), save)
