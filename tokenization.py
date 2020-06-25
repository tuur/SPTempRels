from __future__ import print_function, division
import  os
from nltk.tag import StanfordPOSTagger
import xml.etree.ElementTree as ET 
import networkx as nx


standford_pos_tagger = None

class Token:
	
	def __init__(self, span, string, index, par=None):
		self.span = span
		self.string = string
		self.next = None
		self.previous = None
		self.index = None
		self.paragraph = par
		self.entity_id = None
		self.pos = 'NEWLINE' if string == '\n' else 'UNK_POS'
	
	def __str__(self):
		return str(self.string)
		
	def set_next(self, next):
		self.next = next
	
	def set_previous(self, previous):
		self.previous = previous
	
	def set_index(self, index):
		self.index = index
		
	def get_string(self):
		return str(self.string)

	def get_index(self):
		return self.index

class Tokenization:

	def __init__(self):
		self.tokens = []
		self.num_tokens = 0
		self.max_character_index = 0
		self.start_index_to_token_index = {}
		self.end_index_to_token_index = {}
		self.paragrah_starts = None
		self.dependencies = nx.DiGraph()

	def __str__(self):
		return str([str(token) for token in self.tokens])

	def assign_tokens_to_entity(self, entity, token_assignment=True):
		entity.tokens = self.tokens_in_span(entity.get_span())
		if token_assignment:
			for token in entity.tokens:
				token.entity_id = entity.ID()
		return entity

	def same_paragraph(self,token1,token2):
		return token1.paragraph == token2.paragraph

	def assign_paragraph_to_entities(self,entities):
		for e in entities:
			e.paragraph = self.get_paragraph(e.get_span())

	def get_paragraph(self, span):
		for i,p in enumerate(self.paragrah_starts):
			if span[0] < p:
				return i
		return len(self.paragrah_starts)
		
	def closest_left_entity(self, entity, distance = True):
		d = 0
		if len(entity.tokens) > 0:
			tok_index = max(entity.tokens[0].index,0)
		else:
			tok_index = self.first_left_from_span(entity.get_span()).index
		for i in range(tok_index,0, -1):
				d += 1
				if self.tokens[i].entity_id and self.tokens[i].entity_id!=entity.ID():
					if distance:
						return self.tokens[i].entity_id, d
					else:
						return self.tokens[i].entity_id
		if distance:
			return None, d
		else:
			return None
		
	def closest_right_entity(self, entity, distance = False):
		d = 0 
		if len(entity.tokens) > 0:
			tok_index = entity.tokens[-1].index
		else:
			tok_index = self.first_right_from_span(entity.get_span()).index
		for i in range(tok_index, self.num_tokens):
			d += 1
			if self.tokens[i].entity_id and self.tokens[i].entity_id!=entity.ID():
				if distance:
					return self.tokens[i].entity_id, d
				else:
					return self.tokens[i].entity_id
		if distance:
			return None, d
		else:
			return None


	def POS_tag(self, pos_model):
		global standford_pos_tagger
		
		if not standford_pos_tagger:
			
			standford_pos_tagger = StanfordPOSTagger(os.getcwd() + '/' + pos_model,os.getcwd() + '/stanford-postagger.jar')



		exceptions = set(['\n','\t'])
		selected_tokens = [t for t in self.tokens if not t.get_string() in exceptions]
		tags = standford_pos_tagger.tag([t.get_string() for t in selected_tokens])
		for token, (string, tag) in zip(selected_tokens,tags):
			token.pos = tag
		print('POS:',len(tags),len(selected_tokens))
		if len(tags)!=len(selected_tokens):
			print('POS ERROR: number of tokens, and POS-tags does not match.')
			exit()
	
	def first_left_from_span(self, span):
		for i in range(span[0],0,-1)	:
			if i in self.end_index_to_token_index:
				return self.tokens[self.end_index_to_token_index[i]]
		return Token((0,0),'<start_of_text>')
	
	def first_right_from_span(self, span):
		for i in range(span[1],self.max_character_index):
			if i in self.start_index_to_token_index:
				return self.tokens[self.start_index_to_token_index[i]]
		return Token((0,0),'<end_of_text>')
		
	def n_right_tokens(self, entity, n):	
		n_tokens = []
		if entity.tokens == []:
			tok = self.first_right_from_span(entity.get_span())
		else:
			tok = entity.tokens[0]
		n_tokens = []
		for i in range(1,n):
			tok = self.tokens[tok.next] if tok.next != None else None
			if tok:
				n_tokens.append(tok)
			else:
				return n_tokens
		return n_tokens		
		
	def n_left_tokens(self, entity, n):
		n_tokens = []
		if entity.tokens == []:
			tok = self.first_left_from_span(entity.get_span())
		else:
			tok = entity.tokens[0]
		n_tokens = []
		for i in range(1,n):
			tok = self.tokens[tok.previous] if tok.previous != None else None
			if tok:
				n_tokens.append(tok)
			else:
				return n_tokens
		return n_tokens
			
	def first_left_verb(self, entity):
		token = self.first_left_from_span(entity.get_span())
		d = 0
		while(token and token.paragraph == entity.paragraph):
			if token.pos[0] == 'V':
				return token,d
			token = self.tokens[token.previous] if token.previous != None else None
			d+=1
		return Token((0,0),'NOVERB',-1), 100000000
		
	def first_right_verb(self, entity):
		token = self.first_right_from_span(entity.get_span())
		d = 0
		while(token and token.paragraph == entity.paragraph):
			if token.pos[0] == 'V':
				return token,d
			token = self.tokens[token.next] if token.next != None else None
			d+=1

		return Token((0,0),'NOVERB',-1), 100000000
			
	def assign_tokens_to_entities(self, entities):
		for entity in entities:
			self.assign_tokens_to_entity(entity)

	def tokens_in_span(self, span):
		start, end = None, None
		for i in range(span[0],span[1]):
			if start==None and i in self.start_index_to_token_index:
				start = i
		if start == None:
			return []
		for i in range(span[1],span[0],-1):
			if end==None and i in self.end_index_to_token_index:
				end = i
		if end == None:
			return []
		return self.tokens[self.start_index_to_token_index[start]:self.end_index_to_token_index[end]+1]
		
		
	def tokens_inbetween(self, eLink):
		if not  eLink.get_e1().tokens or not  eLink.get_e2().tokens:
			if eLink.get_e1().get_span()[0] < eLink.get_e2().get_span()[0]:
				return self.tokens_in_span((eLink.get_e1().get_span()[-1], eLink.get_e2().get_span()[0]))				
			else:
				return reversed(self.tokens_in_span((eLink.get_e2().get_span()[-1],eLink.get_e2().get_span()[0])))
		if eLink.get_e1().tokens[0].index < eLink.get_e2().tokens[0].index:
			return self.tokens[eLink.get_e1().tokens[-1].index + 1:eLink.get_e2().tokens[0].index]
		elif eLink.get_e1().tokens[0].index >= eLink.get_e2().tokens[0].index:
			return self.tokens[eLink.get_e1().tokens[0].index + 1:eLink.get_e2().tokens[-1].index:-1]

	def token_distance_between_entities(self, e1, e2):
		if not e1.tokens or not e2.tokens:
			return self.token_distance_between_spans(e1.get_span(), e2.get_span())
		elif e1.tokens[0].index <= e2.tokens[0].index:
			return e2.tokens[0].index - e1.tokens[-1].index
		else:
			return e2.tokens[-1].index - e1.tokens[0].index
		
	def set_paragraph_starts(self, par_starts):
		self.paragrah_starts = par_starts
		
	def token_distance_between_spans(self, span1, span2):
		if span1[0] < span2[0]:
			s1,s2 = span1[-1],span2[0]
			rev = 1
		else:
			s2,s1 = span1[0],span2[-1]+1
			rev = -1
			
		start, end = None, None
		for i in range(s1,s2):
			if start==None and i in self.start_index_to_token_index:
				start = i
		if start == None:
			return 0
		for i in range(s2,s1,-1):
			if end==None and i in self.end_index_to_token_index:
				end = i
		if end == None:
			return 0
		return (self.end_index_to_token_index[end]+2 - self.start_index_to_token_index[start]) * rev
		
		
	def append(self, token):
		self.tokens.append(token)
		self.start_index_to_token_index[token.span[0]] = self.num_tokens
		self.end_index_to_token_index[token.span[1]] = self.num_tokens
		token.set_index(self.num_tokens)
		self.num_tokens += 1
		self.max_character_index = token.span[1]
		
	def read_ctakes(self, doc_id, ctakes_out_dir):
		print('reading ctakes:',doc_id)
		tree = ET.parse(ctakes_out_dir + doc_id + '.xml')
		root = tree.getroot()

		ctakes_POS = {} # from id to POS
		ctakes_DEP = {} # from id to (id, label)
		ctakes_span_to_id = {} # from span to id	
		span_to_index = {}
		for tok in self.tokens:
			span_to_index[tok.span] = tok.index

		# reading xml file			
		for dep in root.iter('org.apache.ctakes.typesystem.type.syntax.ConllDependencyNode'):
			if 'deprel' in dep.attrib:
				label = dep.attrib['deprel']
				head_id = int(dep.attrib['_ref_head'])
				node_id = int(dep.attrib['_id'])
				span = (int(dep.attrib['begin']), int(dep.attrib['end']))
				pos = dep.attrib['postag']
				ctakes_span_to_id[span] = node_id
				ctakes_POS[node_id] = pos
				ctakes_DEP[node_id] = (head_id, label)
		
		# transforming ctakes ids to tokenization ids (token indices)
		ctakes_id_to_token_id = {}
		for span,id in ctakes_span_to_id.items():
			token_index = None
			if span in span_to_index:
				token_index = span_to_index[span]
			elif span[1] in self.end_index_to_token_index:
				token_index = self.end_index_to_token_index[span[1]]
			ctakes_id_to_token_id[id] = token_index
		
		# assigning POS
		for id,pos in ctakes_POS.items():
			if ctakes_id_to_token_id[id]:
				self.tokens[ctakes_id_to_token_id[id]].pos = pos
				
		# assigning dependency relations
		for id, (head_id, label) in ctakes_DEP.items():
			if ctakes_id_to_token_id[id] and head_id in ctakes_id_to_token_id and ctakes_id_to_token_id[head_id]:
				self.dependencies.add_edge(ctakes_id_to_token_id[id], ctakes_id_to_token_id[head_id], label=label + '>')
				self.dependencies.add_edge(ctakes_id_to_token_id[head_id], ctakes_id_to_token_id[id], label='<' + label)



class SimpleTokenizer:
	inclusive_splitters = set([',','.','/','\\','"','\n','=','+','-',';',':','(',')','!','?',"'",'<','>','%','&','$','*','|','[',']','{','}'])
	exclusive_splitters = set([' ','\t'])
	paragraph_splitters = set(['\n\n'])
	
	def tokenize(self, text):
		mem = ""
		start = 0
		paragraph_starts = [0]
		par_mem = " "
		tokens = Tokenization()
		for i,char in enumerate(text):
			par_mem = par_mem[-1] + char			
			if par_mem in self.paragraph_splitters:
				paragraph_starts.append(i)
			
			if char in self.inclusive_splitters:
				if mem!="":
					tokens.append(Token((start,i),text[start:i],len(tokens.tokens), par=len(paragraph_starts)))
					mem = ""
				tokens.append(Token((i,i+1),text[i:i+1],len(tokens.tokens), par=len(paragraph_starts)))
				start = i+1
			elif char in self.exclusive_splitters:
				if mem!="":
					tokens.append(Token((start,i),text[start:i],len(tokens.tokens), par=len(paragraph_starts)))
					mem = ""
					start = i+1
				else:
					start = i+1
			else:
				mem += char
		
		tokens.set_paragraph_starts(paragraph_starts)
		for i,t in enumerate(tokens.tokens):
			if i+1 < len(tokens.tokens):
				t.next = tokens.tokens[i+1].index
				tokens.tokens[i+1].previous = t.index
		return tokens

def test():
	tokenizer = SimpleTokenizer()
	text = "Over de vroege geschiedenis van Rome is veel geschreven."
	tokenization = tokenizer.tokenize(text)
	print([(t.span[0],t.span[1]) for t in tokenization.tokens])
	print(tokenization)
	print([str(t) for t in tokenization.tokens_in_span((0,15))])
