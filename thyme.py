from __future__ import print_function, division

import networkx as nx
from entities import Entity, TLink
import xml.etree.ElementTree as ET 
import glob, re
from collections import OrderedDict as oDict
from tokenization import SimpleTokenizer
from itertools import chain
from copy import copy
import os,shutil
from xml.dom import minidom

class Document:

	def __init__(self, id, txt_file, xml_file, closure=[], lowercase=False, conflate_digits=False, ctakes_out_dir = False, pos=True, less_strict=False, pos_model='english-bidirectional-distsim.tagger'):
			self.txt_file = txt_file
			self.xml_file = xml_file
			self.id = id
			if less_strict:
				return
			self.text = self.read_txt(lowercase=lowercase, conflate_digits=conflate_digits)
			self.events, self.timex3, self.tlinks = oDict(), oDict(),oDict()
			self.read_annotations()
			self.tokenizer = SimpleTokenizer()
			self.tokenization = self.tokenizer.tokenize(self.text)
			self.tokenization.assign_tokens_to_entities(self.events.values())
			self.tokenization.assign_tokens_to_entities(self.timex3.values())	
			self.tokenization.assign_paragraph_to_entities(self.events.values())
			self.tokenization.assign_paragraph_to_entities(self.timex3.values())
			if ctakes_out_dir:
				self.tokenization.read_ctakes(self.id, ctakes_out_dir)
			elif pos:
				self.tokenization.POS_tag(pos_model)
			for label in closure:
				self.closure(label)
			print('TLINKS:',len(self.tlinks))
			self.link_events()
			self.link_entities()
			self.doc_struct = None
			self.extra_events = {}

	def ID(self):
		return self.id
	
	def __str__(self):
		return self.id
	
	def read_txt(self, lowercase=False, conflate_digits=False):
		with open(self.txt_file, 'r') as f:
			text = f.read()
			if lowercase:
				text = text.lower()
			if conflate_digits:
				text = re.sub('\d','5',text)		
			return text

	def get_num_entities_ib(self, e1, e2):
		first, second = sorted([e1,e2],key=lambda e: e.get_span()[0])
		num_events = 0
		num_timex3 = 0
		try:
			while(first.next_entity.get_span() != second.get_span() and first.get_span()[0] <= second.get_span()[0]):
				if first.next_entity.ID() in self.events:
					num_events += 1
					first = first.next_entity
				
				elif first.next_entity.ID() in self.timex3:
					num_timex3 += 1
					first = first.next_entity
		except:
			print('WARNING: Could not get entities inbetween', first, first.get_span(), first.next_entity,'<>', second, second.get_span())
		return num_events, num_timex3	

	def link_events(self):
		sorted_events = sorted(self.events.values(), key=lambda x: x.get_span()[0])
		for i,e in enumerate(sorted_events[:-1]):
			e.next_event = sorted_events[i+1]

	def link_entities(self):
		sorted_entities = sorted(self.events.values() + self.timex3.values(), key=lambda x: x.get_span()[0])
		for i,e in enumerate(sorted_entities[:-1]):
			e.next_entity = sorted_entities[i+1]
	
	def read_annotations(self):
		tree = ET.parse(self.xml_file)
		root = tree.getroot()

		# Reading Entities and DocTimeRel
		for e in root.iter('entity'):
			e_type, e_id, string, span, text_id, doctimerel, e_subtype, e_degree, e_polarity, e_ContextualModality,e_Class = None, None, None, None, self.id, None, None, None, None, None, None
			for child in e.getchildren():
				if child.tag == 'id':
					e_id = child.text
				if child.tag == 'span':
					span = [(int(s1),int(s2)) for (s1,s2) in  [s.split(',') for s in child.text.split(';')]]
					string = ' '.join(self.text[s1:s2] for (s1,s2) in span)
				if child.tag == 'type':
					e_type = child.text
				if child.tag == 'properties':
					for doctime_child in child.iter('DocTimeRel'):
						doctimerel = doctime_child.text
					for e_subtype_child in child.iter('Type'):
						e_subtype = e_subtype_child.text
					for e_degree_child in child.iter('Degree'):
						e_degree = e_degree_child.text
					for e_polarity_child in child.iter('Polarity'):
						e_polarity = e_polarity_child.text
					for e_ContextualModality_child in child.iter('ContextualModality'):
						e_ContextualModality = e_ContextualModality_child.text						
					for e_Class_child in child.iter('Class'):
						e_Class = e_Class_child.text	
			
			if e_type == 'EVENT' and doctimerel:
				self.events[e_id] = Entity(e_type, e_id, string, span, text_id, doctimerel,etree=e)
				self.events[e_id].attributes = {'EVENT_Type':e_subtype, 'EVENT_Polarity':e_polarity,'EVENT_Degree':e_degree, 'EVENT_CONTEXTUAL_MODALITY':e_ContextualModality}

			if e_type in set(['TIMEX3','SECTIONTIME','DOCTIME']):
				self.timex3[e_id] = Entity(e_type, e_id, string, span, text_id, doctimerel,etree=e)
				self.timex3[e_id].attributes = {'TIMEX_Class':e_Class}

		# Reading Relations (TLINKS)
		for r in root.iter('relation'):
			source, target, relation = None, None, None
			for child in r.getchildren():
				if child.tag == 'properties':
					for properties_child in child:
						if properties_child.tag == 'Source':
							if properties_child.text in self.events:
								source = self.events[properties_child.text]
							elif properties_child.text in self.timex3:
								source = self.timex3[properties_child.text]
						if properties_child.tag == 'Target':
							if properties_child.text in self.events:
								target = self.events[properties_child.text]
							elif properties_child.text in self.timex3:
								target = self.timex3[properties_child.text]	
						if properties_child.tag == 'Type':
							relation = properties_child.text

			if relation in set(['CONTAINS', 'BEFORE', 'OVERLAP','BEGINS-ON','ENDS-ON']):
				tlink_id = source.ID() + '-' + target.ID()
				self.tlinks[tlink_id] = TLink(source, target, relation)

		print(self.id,'\t','events:',len(self.events), '\ttimex3:', len(self.timex3), 'tlink:', len(self.tlinks))

	#@profile
	def get_doctimerel_candidates(self):
		return [e for e_id,e in self.events.items()]

	#@profile	
	def get_tlink_candidates(self, labels, max_token_distance=None, same_par=None):
		candidates = []
		e1s = chain(self.events.values(), self.timex3.values())
		e2s =  self.events.values()	
			
		for e1 in e1s:
			for e2 in e2s:
				if max_token_distance and abs(self.tokenization.token_distance_between_entities(e1,e2)) > max_token_distance:
					continue
				if same_par and not self.tokenization.same_paragraph(e1,e2):
					continue
				if e1 == e2 or e1.get_span() == e2.get_span():
					continue
				if e1.ID() + '-' + e2.ID() in self.tlinks and self.tlinks[e1.ID() + '-' + e2.ID()].get_tlink() in labels:
					candidates.append(self.tlinks[e1.ID() + '-' + e2.ID()])					
				else:
					candidates.append(TLink(e1,e2,'no_label'))
					
		print('max_recall', float(len([l for l in self.tlinks.values() if l in candidates])) / (len(self.tlinks.values()) + 0.00001), len(self.tlinks))
		return candidates		

	def closure(self,label):
		pairs = [(tl.get_e1(),tl.get_e2()) for tl in self.tlinks.values() if tl.tlink == label]	
		G = nx.DiGraph(list(pairs))
		h = nx.DiGraph([(u,v,{'d':l}) for u,adj in nx.floyd_warshall(G).items() for v,l in adj.items() if l > 0 and l < float('inf')])
		new_pairs = set([ed for ed in h.edges() if not ed in G.edges()])
		print('closure:',label, len(new_pairs))
		for (source, target) in new_pairs:
			tlink_id = source.ID() + '-' + target.ID()
			self.tlinks[tlink_id] = TLink(source, target, label)



def read_thyme(thyme_path, regex='.*Temp.*', max_documents=None, closure=[], lowercase=False, conflate_digits=False, ctakes_out_dir=False, pos=True, pos_model='english-bidirectional-distsim.tagger', datasets=['Train','Dev','Test']):
	document_structure = {}
	for dataset in datasets:
		print('<<',dataset,'>>')
		document_structure[dataset] = read_thyme_documents(thyme_path + '/' + dataset, regex, max_documents, closure, lowercase, conflate_digits, ctakes_out_dir=ctakes_out_dir,pos=pos, pos_model=pos_model)	
	return document_structure

def read_thyme_documents(folder, regex, max_documents, closure=[], lowercase=False, conflate_digits=False, ctakes_out_dir=False, pos=True, less_strict=False, pos_model='english-bidirectional-distsim.tagger'):
        documents = []
        for i,subfolder_path in enumerate(glob.glob(folder + '/*')):
                subfolder_name = subfolder_path.split('/')[-1]

                text_file, annotations_file = '', ''
                for file_path in glob.glob(subfolder_path + '/*'):
                        file_name = file_path.split('/')[-1]
                        if file_name == subfolder_name:
                                text_file = file_path
                        if re.search(regex, file_name):
                                annotations_file = file_path
                if (text_file and annotations_file) or less_strict:
                    documents.append(Document(text_file.split('/')[-1], text_file, annotations_file, closure, lowercase=lowercase,conflate_digits=conflate_digits, ctakes_out_dir=ctakes_out_dir, pos=pos, less_strict=less_strict, pos_model=pos_model))
                    if max_documents and i >= max_documents:
                        return documents
                else:
                        print('warning: no annotations or text for',subfolder_name, '(therefore skipped)')

        return documents


def write_to_anafora(X, preds, output_dir, document_structure):
	for Yp, Y_name in preds:
		pred_dir = output_dir + '/' + Y_name
		print('writing',Y_name,'to', pred_dir)

		if os.path.exists(pred_dir):
			shutil.rmtree(pred_dir)
		os.makedirs(pred_dir)
			
		for doc in document_structure:
			doc_dir = pred_dir + '/' + doc.ID()
			new_doc_file = doc_dir + '/' + doc.ID() +'.Temporal-Relation.system.completed.xml'
			os.makedirs(doc_dir)
			
			
			doc_xml = ET.Element('data')
			doc_xml_annotations = ET.SubElement(doc_xml,'annotations')	
			
			for id,xt in doc.timex3.items():
					doc_xml_annotations.append(xt.get_etree())
					
			for ((X_e, X_ee),(Y_e,Y_ee)) in zip(X,Yp):
				for i,xe in enumerate(X_e):
					if xe.get_doc_id() == doc.ID():
						xe_new_etree = copy(xe.get_etree())
						if len(list(xe_new_etree.iter('DocTimeRel'))) == 0:
							for p in xe_new_etree.iter('properties'):
								ET.SubElement(p, 'DocTimeRel')
						
						for dct in xe_new_etree.iter('DocTimeRel'):
							dct.text = Y_e[i]
						doc_xml_annotations.append(xe_new_etree)
						
					
				for i,(xee,yee) in enumerate([(xee,Y_ee[j]) for j,xee in enumerate(X_ee) if Y_ee[j]!='no_label']):
					if xee.get_e1().get_doc_id() == doc.ID():
						xee_new = ET.Element('relation')
						
						xee_new_id = ET.SubElement(xee_new,'id')
						xee_new_id.text = str(i) + '@r@' + xee.get_e1().get_doc_id() + '@system'
						
						xee_new_type = ET.SubElement(xee_new,'type')
						xee_new_type.text = 'TLINK'
						
						xee_new_parentstype = ET.SubElement(xee_new,'parentsType')
						xee_new_parentstype.text = 'TemporalRelations'
						
						xee_new_props = ET.SubElement(xee_new,'properties')
						
						xee_new_source = ET.SubElement(xee_new_props, 'Source')
						xee_new_source.text = xee.get_e1().ID()
						
						xee_new_reltype = ET.SubElement(xee_new_props, 'Type')
						xee_new_reltype.text = yee
						
						xee_new_target = ET.SubElement(xee_new_props, 'Target')
						xee_new_target.text = xee.get_e2().ID()

						doc_xml_annotations.append(xee_new)

			doc_xml_string = minidom.parseString(ET.tostring(doc_xml).replace('\n','').replace('\t', '')).toprettyxml(indent = "\t", newl='\n\n')
			with open(new_doc_file, 'w') as f:
				print('writing',new_doc_file,'...')
				f.write(doc_xml_string)
			


