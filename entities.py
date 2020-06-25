from __future__ import print_function, division

class Entity(object):

	def __init__(self, type, id, string, spans, text_id, doctimerel=None, etree=None):
		self.type = type
		self.id = id
		self.string = string
		self.spans = spans
		self.text_id = text_id
		self.doctimerel = doctimerel
		self.phi = {}
		self.phi_v = None
		self.tokens = None
		self.paragraph = None
		self.xmltree = etree
		self.embedding = None
		self.next_event = None
		self.next_entity = None
		self.attributes = {}
	
	def __str__(self):
		return str(self.string)

	def __hash__(self):
		return hash(self.id)
	
	def __eq__(self, other):
		return self.id == other.id		

	def __ne__(self, other):
		return not(self == other)

	def type(self):
		return self.type

	def ID(self):
		return self.id
		
	def get_tokens(self):
		return self.tokens

	def text_id(self):
		return self.text_id

	def get_doctimerel(self):
		return self.doctimerel
		
	def get_span(self):
		return self.spans[0]
	
	def get_etree(self):
		return self.xmltree
		
	def get_doc_id(self):
		return self.id.split('@')[2]		
		
class TLink(object):

	def __init__(self, e1, e2, tlink=None):
		self.e1 = e1
		self.e2 = e2
		self.tlink = tlink
		self.phi = {}
		self.phi_v = None
		self.tokens_ib = None
		self.id = None
		
	def __str__(self):
		return str(self.e1) + '-' + str(self.e2)

	def ID(self):
		if not self.id:
			self.id = self.e1.ID() + '-' + self.e2.ID()
		return self.id

	def __hash__(self):
		return hash(self.id())
	
	def __eq__(self, other):
		return self.ID() == other.ID()	

	def __ne__(self, other):
		return not (self == other)
		
	def set_tokens_ib(self, tokens):
		self.tokens_ib = list(tokens)

	def get_tokens_ib(self):
		return self.tokens_ib

	def type(self):
		return self.e1.type + '-' + self.e2.type

	def get_tlink(self):
		return self.tlink
	
	def get_e1(self):
		return self.e1
	
	def get_e2(self):
		return self.e2


