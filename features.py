from __future__ import print_function, division
from sklearn.feature_extraction import DictVectorizer
import networkx as nx


class FeatureHandler(object):
	
	def __init__(self, sparse=True):
		self.feature_template = {}
		self.feature_vectorizer = DictVectorizer(sparse=sparse)
		self.dim  = None
		self.token_clusters = {}
		
		
	def read_token_clusters(self, clusterfile):
		clustering_name = clusterfile.split('/')[-1]
		self.token_clusters[clustering_name] = {}
		with open(clusterfile, 'r') as f:
			for line in f.readlines():
				token, cluster = line.rstrip().split('\t')
				if token == '<newline>':
					token = '\n'
				self.token_clusters[clustering_name][token] = cluster
					
	def update_vectorizer(self):
		self.feature_vectorizer.fit([self.feature_template])
		self.dim = len(self.feature_vectorizer.get_feature_names())
	
	def vectorize(self, feature_dict):
		return self.feature_vectorizer.transform(feature_dict)
		
	def inverse_transform(self, vector):
		return self.feature_vectorizer.inverse_transform(vector)
		
	def feature_names(self):
		return self.feature_template.keys()
					

class StructuredFeatureHandler(FeatureHandler):
	DCTR_bigrams = False
	DCTR_trigrams = False
	DCTR_of_TLINKS = False
	TLINK_argument_bigrams = False
	
	
	def __init__(self, labels_e, labels_ee):
		super(StructuredFeatureHandler, self).__init__()
		self.labels_e, self.labels_ee = labels_e, labels_ee
	
	def extract(self, X, Y=None, update=True):
		
		X_e, X_ee = X
		if Y:
			Y_e, Y_ee = Y
		X_e_ids = {e.ID():i for i,e in enumerate(X_e)}
		X_ee_ids = {ee.ID():i for i,ee in enumerate(X_ee)}
		features, features_expressions = {},{}

		
		if self.DCTR_bigrams:
			if not Y:
				features_expressions.update({'DCTR_bigram:'+str((l1,l2)):[] for l1 in self.labels_e for l2 in self.labels_e})
			else:
				features.update({'DCTR_bigram:'+str((l1,l2)):0 for l1 in self.labels_e for l2 in self.labels_e})
			for e in X_e:
				if e.type == 'EVENT' and  e.next_event and e.next_event.ID() in X_e_ids and e.paragraph == e.next_event.paragraph:
					if not Y:
						for l1 in self.labels_e:
							for l2 in self.labels_e:
								features_expressions['DCTR_bigram:' + str((l1,l2))].append(('Ie:' + e.ID() +':' + l1, 'Ie:' + e.next_event.ID() +':' + l2))
					else:
						features['DCTR_bigram:' + str((Y_e[X_e_ids[e.ID()]], Y_e[X_e_ids[e.next_event.ID()]]))] += 1		


		if self.DCTR_trigrams:
			if not Y:
				features_expressions.update({'DCTR_trigram:'+str((l1,l2,l3)):[] for l1 in self.labels_e for l2 in self.labels_e for l3 in self.labels_e})
			else:
				features.update({'DCTR_trigram:'+str((l1,l2,l3)):0 for l1 in self.labels_e for l2 in self.labels_e for l3 in self.labels_e})
			for e in X_e:	
				if e.type == 'EVENT' and  e.next_event and e.next_event.ID() in X_e_ids and e.paragraph == e.next_event.paragraph and  e.next_event.next_event and e.next_event.next_event.ID() in X_e_ids and e.paragraph == e.next_event.next_event.paragraph:
					if not Y:
						for l1 in self.labels_e:
							for l2 in self.labels_e:
								for l3 in self.labels_e:
									features_expressions['DCTR_trigram:' + str((l1,l2,l3))].append(('Ie:' + e.ID() +':' + l1, 'Ie:' + e.next_event.ID() +':' + l2, 'Ie:' + e.next_event.next_event.ID() +':' + l3))
					else:
						features['DCTR_trigram:' + str((Y_e[X_e_ids[e.ID()]], Y_e[X_e_ids[e.next_event.ID()]], Y_e[X_e_ids[e.next_event.next_event.ID()]]))] += 1		

	
		if self.DCTR_of_TLINKS:
			if not Y:
				features_expressions.update({'DCTR_TLINKS:' + rel + ':' + str((l1,l2)):[] for l1 in self.labels_e for l2 in self.labels_e for rel in self.labels_ee if not rel == 'no_label'})
			else:
				features.update({'DCTR_TLINKS:' + rel + ':' + str((l1,l2)):0 for l1 in self.labels_e for l2 in self.labels_e for rel in self.labels_ee if not rel == 'no_label'})
			for ePair in X_ee:
				if ePair.get_e1().ID() in X_e_ids and ePair.get_e2().ID() in X_e_ids:
					if not Y:
						for rel in self.labels_ee:
							if not rel == 'no_label':
								for l1 in self.labels_e:
									for l2 in self.labels_e:
										features_expressions['DCTR_TLINKS:' + rel + ':' + str((l1,l2))].append(('Iee:' + ePair.ID() + ':' + rel, 'Ie:' + ePair.get_e1().ID() +':' + l1, 'Ie:' + ePair.get_e2().ID() +':' + l2))
					elif not ePair.tlink == 'no_label':
						features['DCTR_TLINKS:' +ePair.tlink + ':' + str((Y_e[X_e_ids[ePair.get_e1().ID()]], Y_e[X_e_ids[ePair.get_e2().ID()]]))] += 1		
	
		if self.TLINK_argument_bigrams:
						
			if Y:
				features.update({'TLINK_arg_bigrams:' + str(((t1,a1,l1),(t2,a2,l2))) :1 for l1 in self.labels_ee for l2 in self.labels_ee for a1 in ['arg1','arg2'] for a2 in ['arg1','arg2'] for t1 in ['TIMEX3','EVENT','SECTIONTIME','DOCTIME'] for t2 in ['TIMEX3','EVENT','SECTIONTIME','DOCTIME']})
				roles = {}
				for i,tlink in enumerate(X_ee):

					if tlink.get_e1() in roles:
						roles[tlink.get_e1()].append((tlink.get_e1().type, 'arg1',Y_ee[i]))
					else:
						roles[tlink.get_e1()] = [(tlink.get_e1().type, 'arg1',Y_ee[i])]
					
					if tlink.get_e2() in roles:
						roles[tlink.get_e2()].append((tlink.get_e2().type, 'arg2',Y_ee[i]))
					else:
						roles[tlink.get_e2()] = [(tlink.get_e2().type, 'arg2',Y_ee[i])]	

				for e in roles:
					if e.next_entity and e.next_entity in roles and e.next_entity.paragraph == e.paragraph:
						for r_e1 in roles[e]:
							for r_e2 in roles[e.next_entity]:
								features['TLINK_arg_bigrams:' + str((r_e1,r_e2))] += 1	
			
			else:
				arg1s = set([tlink.get_e1() for tlink in X_ee])
				entities = set([tlink.get_e1() for tlink in X_ee] + [tlink.get_e2() for tlink in X_ee])				
				for e in entities:
					if e.next_entity and e.next_entity in entities and e.next_entity.paragraph == e.paragraph:
						for l1 in self.labels_ee:
							for l2 in self.labels_ee:
									a1 = 'arg1' if e in arg1s else 'arg2'
									a2 = 'arg1' if e.next_entity in arg1s else 'arg2'
									if 'TLINK_arg_bigrams:' + str(((e.type,a1,l1), (e.next_entity.type,a2,l2))) in features_expressions:
										features_expressions['TLINK_arg_bigrams:' + str(((e.type,a1,l1), (e.next_entity.type,a2,l2)))].append((e.ID() + ':' + a1 +':' + l1, e.next_entity.ID() + ':' + a2 +':' + l2))			
									else:
										features_expressions['TLINK_arg_bigrams:' + str(((e.type,a1,l1), (e.next_entity.type,a2,l2)))]= [(e.ID() + ':' + a1 +':' + l1, e.next_entity.ID() + ':' + a2 +':' + l2)]
					
			
		
		if update:
			self.feature_template.update(features)

		if not Y:
			return features_expressions
		else:
			return features

	
class DocTimeRelFeatureHandler(FeatureHandler):
	entity_tokens = True
	entity_attributes = True
	entity_pos_tags = True
	character_n_grams = None
	entity_tok_clusters=False

	left_bow_context = [3,5]
	right_bow_context = [3,5]
	left_context_pos_bigrams = [3,5]
	right_context_pos_bigrams = [3,5]
	left_context_cl_bigrams = False
	right_context_cl_bigrams	= False
	
	surrounding_entities = True
	surrounding_entities_pos = True

	closest_entity = False
	closest_verb = True

	
	def extract(self, entity, document, update=True):
		features = {'label_bias:':1}

		if self.entity_tokens:
			features.update({'entity_tokens:' + token.get_string():1 for token in entity.tokens})
			features.update({'entity_string:' + str(entity):1})
		
		if self.entity_pos_tags:
			features.update({'entity_pos:' + str([t.pos for t in entity.tokens]):1})

		if self.entity_tok_clusters:
			for clustering in self.token_clusters:
				features.update({'entity_cl_' + str(clustering) + ':' + str([self.token_clusters[clustering][t.get_string()] for t in entity.tokens]):1})

		if self.left_context_pos_bigrams:
			for window in self.left_context_pos_bigrams:
				features.update({'left_pos_context(' + str(window) + '):' + str(ngram):1 for ngram in get_ngrams([t.pos for t in document.tokenization.n_left_tokens(entity, window)],1)})

		if self.left_context_cl_bigrams:
			for clustering in self.token_clusters:
				for window in self.left_context_cl_bigrams:
					features.update({'left_cl_' + clustering + '_context(' + str(window) + '):' + str(ngram):1 for ngram in get_ngrams([self.token_clusters[clustering][t.get_string()] for t in document.tokenization.n_left_tokens(entity, window)],1)})

		if self.right_context_pos_bigrams:
			for window in self.right_context_pos_bigrams:
				features.update({'right_pos_context(' + str(window) + '):' + str(ngram):1 for ngram in get_ngrams([t.pos for t in document.tokenization.n_right_tokens(entity, window)],1)})

		if self.left_context_cl_bigrams:
			for clustering in self.token_clusters:
				for window in self.left_context_cl_bigrams:
					features.update({'left_cl_' + clustering + '_context(' + str(window) + '):' + str(ngram):1 for ngram in get_ngrams([self.token_clusters[clustering][t.get_string()] for t in document.tokenization.n_right_tokens(entity, window)],1)})


		if self.left_bow_context:
			for window in self.left_bow_context:
				features.update({'left_bow_context(' + str(window) + '):' + token.get_string():1 for token in document.tokenization.n_left_tokens(entity, window)})

		if self.right_bow_context:
			for window in self.right_bow_context:
				features.update({'right_bow_context(' + str(window) + '):' + token.get_string():1 for token in document.tokenization.n_right_tokens(entity, window)})

		if self.entity_attributes:
			features.update({attr +':' + str(entity.attributes[attr]):1 for attr in entity.attributes})
		
		if self.surrounding_entities or self.closest_entity or  self.surrounding_entities_pos:
			left_e_id,left_dist = document.tokenization.closest_left_entity(entity, True)
			right_e_id,right_dist =  document.tokenization.closest_right_entity(entity, True)
			if left_e_id:
				left_e = document.events[left_e_id] if left_e_id in document.events else document.timex3[left_e_id]
				if self.surrounding_entities:
					features.update({'left_e:token:' + t.get_string(): 1 for t in left_e.get_tokens()})
				if self.closest_entity and left_dist <= right_dist:
					features.update({'closest_e:token:' + t.get_string(): 1 for t in left_e.get_tokens()})
					if self.surrounding_entities_pos:
						features.update({'closest_entity_pos:' + str([t.pos for t in left_e.tokens]):1})
					
			if right_e_id:
				right_e = document.events[right_e_id] if right_e_id in document.events else document.timex3[right_e_id]
				if self.surrounding_entities:
					features.update({'right_e:token:' + t.get_string(): 1 for t in right_e.get_tokens()})
				if self.closest_entity and right_dist < left_dist:
					features.update({'closest_e:token:' + t.get_string(): 1 for t in right_e.get_tokens()})
					if self.surrounding_entities_pos:
						features.update({'closest_entity_pos:' + str([t.pos for t in right_e.tokens]):1})
					
		if self.character_n_grams:
			features.update({'char_n_grams:' + str(ngram):1 for ngram in get_ngrams('_' + entity.string + '_' ,self.character_n_grams)})
		

		if self.closest_verb:
			closest_left, dl = document.tokenization.first_left_verb(entity)
			
			closest_right, dr = document.tokenization.first_right_verb(entity)
			
			
			closest = closest_left if dl < dr else closest_right
			features.update({'closest_verb:' + closest.get_string():1})
			features.update({'closest_verb_pos:' + closest.pos:1})

			
		if update:
			self.feature_template.update(features)
			
		return features
		
		
class TLinkFeatureHandler(FeatureHandler):
	entity_tokens = True
	entity_token_window = 3
	entity_pos_window = 3
	entity_attributes = True
	entity_pos_tags = True


	entity_cls = False	
	entity_cl_window = False
	cl_ngrams_inbetween = False

	num_entities_ib = False
	token_distance = False 

	entity_order = True
	ordered = True
	
	subsequences_inbetween = False

	closest_entities_flag = False
	dep_path = True
	ee_type = True
	ngrams_inbetween = 3	
	pos_ngrams_inbetween = 3
	
	
	def extract(self, entityPair, document, update=True):
		features = {'label_bias:':1}
		
		if self.entity_tokens:				
			features.update({'e1_entity_tokens:' + token.get_string():1 for token in entityPair.get_e1().get_tokens()})
			features.update({'e1_string:' + str(entityPair.get_e1()):1})
			features.update({'e2_entity_tokens:' + token.get_string():1 for token in entityPair.get_e2().get_tokens()})
			features.update({'e2_string:' + str(entityPair.get_e2()):1})
			
		if self.entity_pos_tags:
			features.update({'e1_pos:' + str([t.pos for t in entityPair.get_e1().tokens]):1})			
			features.update({'e2_pos:' + str([t.pos for t in entityPair.get_e2().tokens]):1})			

		if self.entity_cls:
			for clustering in self.token_clusters:
				features.update({'e1_cl_' + clustering + ':' + str([self.token_clusters[clustering][t.get_string()] for t in entityPair.get_e1().tokens]):1})			
				features.update({'e2_' + clustering + ':' + str([self.token_clusters[clustering][t.get_string()] for t in entityPair.get_e2().tokens]):1})	
			
		if self.entity_attributes:
			features.update({'e1_attribute:' + attr + ':' + str(entityPair.get_e1().attributes[attr]):1 for attr in entityPair.get_e1().attributes})
			features.update({'e2_attribute:' + attr + ':' + str(entityPair.get_e2().attributes[attr]):1 for attr in entityPair.get_e2().attributes})

		if self.num_entities_ib:
			num_events, num_timex3 = document.get_num_entities_ib(entityPair.get_e1(), entityPair.get_e2())
			features.update({'num_events_ib:' +str(num_events):1,'num_timex3s_ib:' +str(num_timex3):1})
			
		
		if self.closest_entities_flag:
			left_e_id,left_dist = document.tokenization.closest_left_entity(entityPair.get_e1(), True)
			right_e_id,right_dist =  document.tokenization.closest_right_entity(entityPair.get_e1(), True)
			if left_e_id == entityPair.get_e2().ID() and left_dist < right_dist:
				features.update({'closest_entities_flag':1})
			elif right_e_id == entityPair.get_e2().ID() and right_dist< left_dist:
				features.update({'closest_entities_flag':1})
			
		if self.token_distance:
			dist = document.tokenization.token_distance_between_entities(entityPair.get_e1(), entityPair.get_e2())
			features.update({'token_distance:':dist})

		if self.dep_path:
			graph = document.tokenization.dependencies
			paths = []
			for tok1 in entityPair.get_e1().tokens:
				for tok2 in entityPair.get_e2().tokens:
					if tok1.index in graph.node and tok2.index in graph.node:
						if nx.has_path(graph, tok1.index, tok2.index):
							paths.append(nx.shortest_path(graph, tok1.index, tok2.index))
			if not paths == []:
				for p in paths:
					feature_path = 'dep_path:'
					for i,t in enumerate(p):
						feature_path += '[' + document.tokenization.tokens[t].pos + ']' 
						if i + 1 < len(p):
							label = graph.edge[t][p[i+1]]['label']
							feature_path += label
					features.update({feature_path:1})

			
		if self.ngrams_inbetween or self.pos_ngrams_inbetween:
			if entityPair.tokens_ib == None:
				entityPair.set_tokens_ib(document.tokenization.tokens_inbetween(entityPair))			
			if self.ordered:
				if self.ngrams_inbetween:
					for n in range(1,self.ngrams_inbetween + 1):
						features.update({'reversed:' + str(entityPair.e2.get_span()[0] < entityPair.e1.get_span()[0]) + ':' + str(n) + '-grams_inbetween:' + str(ngram):1 for ngram in get_ngrams([t.get_string() for t in entityPair.get_tokens_ib()],n)})					
				if self.pos_ngrams_inbetween:		
					for n in range(1,self.pos_ngrams_inbetween + 1):
						features.update({'reversed:' + str(entityPair.e2.get_span()[0] < entityPair.e1.get_span()[0]) + ':pos_' + str(n) + '-grams_inbetween:' + str(ngram):1 for ngram in get_ngrams([t.pos for t in entityPair.get_tokens_ib()],n)})
				if self.cl_ngrams_inbetween:
					for n in range(1,self.cl_ngrams_inbetween + 1):
						features.update({'reversed:' + str(entityPair.e2.get_span()[0] < entityPair.e1.get_span()[0]) + ':' + clustering +'_' + str(n) + '-grams_inbetween:' + str(ngram):1 for ngram in get_ngrams([self.token_clusters[clustering][t.get_string()]for t in entityPair.get_tokens_ib()],n)})
			
			else:
				if self.ngrams_inbetween:
					for n in range(1,self.ngrams_inbetween + 1):
						features.update({str(n) + '-grams_inbetween:' + str(ngram):1 for ngram in get_ngrams([t.get_string() for t in entityPair.get_tokens_ib()],n)})					
				if self.pos_ngrams_inbetween:		
					for n in range(1,self.pos_ngrams_inbetween + 1):
						features.update({'pos_' + str(n) + '-grams_inbetween:' + str(ngram):1 for ngram in get_ngrams([t.pos for t in entityPair.get_tokens_ib()],n)})
				if self.cl_ngrams_inbetween:
					for n in range(1,self.cl_ngrams_inbetween + 1):
						features.update({clustering +'_' + str(n) + '-grams_inbetween:' + str(ngram):1 for ngram in get_ngrams([self.token_clusters[clustering][t.get_string()]for t in entityPair.get_tokens_ib()],n)})

						
		if self.entity_order:
			features.update({'e1>e2': int(entityPair.e1.get_span()[0] < entityPair.e2.get_span()[0]) , 'e1<e2':int(entityPair.e1.get_span()[0] > entityPair.e2.get_span()[0])})
		
		if self.subsequences_inbetween:
			if entityPair.tokens_ib == None:
				entityPair.set_tokens_ib(document.tokenization.tokens_inbetween(entityPair))
			features.update({'subseq_inbetween:' + t1.get_string() +'->'+t2.get_string():1 for (t1,t2) in subpairs(entityPair.get_tokens_ib())})
				
	
		if self.entity_token_window:
			features.update({'e1_right_token_window:' + t.get_string():1 for t in document.tokenization.n_right_tokens(entityPair.get_e1(), self.entity_token_window)})
			features.update({'e2_right_token_window:' + t.get_string():1 for t in document.tokenization.n_right_tokens(entityPair.get_e2(), self.entity_token_window)})				
			features.update({'e1_left_token_window:' + t.get_string():1 for t in document.tokenization.n_left_tokens(entityPair.get_e1(), self.entity_token_window)})
			features.update({'e2_left_token_window:' + t.get_string():1 for t in document.tokenization.n_left_tokens(entityPair.get_e2(), self.entity_token_window)})				

		if self.entity_pos_window:
			features.update({'e1_right_pos_window:' + t.pos:1 for t in document.tokenization.n_right_tokens(entityPair.get_e1(), self.entity_pos_window)})
			features.update({'e2_right_pos_window:' + t.pos:1 for t in document.tokenization.n_right_tokens(entityPair.get_e2(), self.entity_pos_window)})				
			features.update({'e1_left_pos_window:' + t.pos:1 for t in document.tokenization.n_left_tokens(entityPair.get_e1(), self.entity_pos_window)})
			features.update({'e2_left_pos_window:' + t.pos:1 for t in document.tokenization.n_left_tokens(entityPair.get_e2(), self.entity_pos_window)})				

		if self.entity_cl_window:
			for clustering in self.token_clusters:
				features.update({'e1_right_'+clustering+'_window:' + self.token_clusters[clustering][t.get_string()]:1 for t in document.tokenization.n_right_tokens(entityPair.get_e1(), self.entity_cl_window)})
				features.update({'e2_right_'+clustering+'_window:' + self.token_clusters[clustering][t.get_string()]:1 for t in document.tokenization.n_right_tokens(entityPair.get_e2(), self.entity_cl_window)})				
				features.update({'e1_left_'+clustering+'_window:' + self.token_clusters[clustering][t.get_string()]:1 for t in document.tokenization.n_left_tokens(entityPair.get_e1(), self.entity_cl_window)})
				features.update({'e2_left_'+clustering+'_window:' + self.token_clusters[clustering][t.get_string()]:1 for t in document.tokenization.n_left_tokens(entityPair.get_e2(), self.entity_cl_window)})				
				
		
		if self.ee_type:
			features.update({'ee_type:' + str(entityPair.type()): 1})

		
		if update:
			self.feature_template.update(features)

		return features

def get_ngrams(sequence, n):
		ngrams = []	
		for i in range(0,len(sequence)-n):
			ngrams.append(sequence[i:i+n])
		return ngrams
		
def subpairs(sequence, window_size = 2):
	sequence = list(sequence)
	subsequences = []
	for i in range(len(sequence) - window_size):
		for j in range(i+1,i + window_size):
			subsequences.append((sequence[i],sequence[j]))
	return subsequences

