from __future__ import print_function, division
import gurobipy as grb
import numpy as np
from scipy.sparse import csr_matrix
import random
import pickle
random.seed(1)


class StructuredPerceptron:
	
	
	def __init__(self, labels, dims, feature_extractor_e,feature_extractor_ee, averaged=False, loss_augmented_training=False, balance=False, structured_feature_handler=None, regularization=False, regularization_term=1.0):
		self.labda_e, self.labda_ee, self.labda_struct = {l:np.ones(dims[0]) for l in labels[0]}, {l:np.ones(dims[1]) for l in labels[1]}, {l:0 for l in structured_feature_handler.feature_names()}
		self.averaged = averaged
		if self.averaged:
			self.labdaC_e, self.labdaC_ee, self.labdaC_struct = {l:np.ones(dims[0]) for l in labels[0]}, {l:np.ones(dims[1]) for l in labels[1]}, {l:0 for l in structured_feature_handler.feature_names()}
		self.loss_augmented_training = loss_augmented_training		
		self.labels_e, self.labels_ee = labels
		self.loss_trajectory = {'e':[],'ee':[]}
		self.dims = {'e':dims[0],'ee':dims[1], 'struct':dims[2]}
		self.balance = balance
		self.structured_feature_handler = structured_feature_handler
		self.regularization = regularization
		self.regularization_term = regularization_term
		self.feature_extractor_e = feature_extractor_e
		self.feature_extractor_ee = feature_extractor_ee
		
	def set_labda(self, new_labda_e, new_labda_ee):
		self.labda_e, self.labda_ee = new_labda_e, new_labda_ee

	def train(self, X, Y, num_iterations=10, constraints = set(['MUL']), learning_rate=1.0, decreasing_lr = False, shuffle=False, negative_subsampling='random', stop_criteria=0, dropout=0.1):
		
		# calculating class balance
		if self.balance: 
			self.balance = {'e':{l:1.0 for l in self.labda_e}, 'ee':{l:1.0 for l in self.labels_ee}}
			e_sum, ee_sum = 0, 0
			for d in Y:
				for ye in d[0]:
					self.balance['e'][ye] += 1
					e_sum +=1
				for yee in d[1]:
					ee_sum +=1
					self.balance['ee'][yee] += 1
			self.balance = {'e':{l:float(self.balance['e'][l]) / e_sum for l in self.labels_e }, 'ee':{l:float(self.balance['ee'][l]) / ee_sum for l in self.labels_ee}}
			print('balance:',self.balance)
		else:
			self.balance = {'e':{l:1.0 for l in self.labda_e}, 'ee':{l:1.0 for l in self.labels_ee}}
		
		# initializing variables
		size = len(X)
		if stop_criteria:
			dev_X, X = X[:int(stop_criteria*size)], X[int(stop_criteria*size):]
			dev_Y, Y = Y[:int(stop_criteria*size)], Y[int(stop_criteria*size):]
			size = len(X)
		
		
		lr = learning_rate
		indices = list(range(size))
		C = 1
		gurobi_models = [None for i in range(size)]
		
		# start of training
		print('Training Structured Perceptron...')
		for i in range(num_iterations):
			if decreasing_lr:
				lr = learning_rate * ((num_iterations - i) / num_iterations)
			if shuffle:
				random.shuffle(indices)
			loss_e, loss_ee = 0,0
			print('--iteration:',i,'\tlr:',lr)
			for j in indices[:int(-1 * dropout * size)]:
				X_k,Y_k = X[j], Y[j]
				if negative_subsampling:
					X_k, Y_k = self.get_negative_sample(X_k,Y_k,typ=negative_subsampling)
					
				gurobi_models[j] = None	
				Y_p, gurobi_models[j] = self.decode(X_k, constraints, loss_augmentation=Y_k, gurobi_model=gurobi_models[j],j=j, gurobi_model_out=True) 
				loss_e_k, loss_ee_k = self.loss_both(Y_p, Y_k)
				if loss_e_k + loss_ee_k > 0:
					self.update(X_k, Y_k, Y_p, C, lr)
				C += 1
				loss_e += loss_e_k
				loss_ee += loss_ee_k

			if stop_criteria:
				devYp = self.predict(dev_X, constraints)
				loss_e, loss_ee = sum([self.loss(ype,ye) for ((ype,ypee),(ye,yee)) in zip(devYp,dev_Y)]),sum([self.loss(ypee,yee) for ((ype,ypee),(ye,yee)) in zip(devYp,dev_Y)])

			self.loss_trajectory['e'].append(loss_e)
			self.loss_trajectory['ee'].append(loss_ee)


			print('avg_loss_e:',loss_e / size, 'avg_loss_ee',loss_ee /size)
			print(len(self.labda_struct), sorted(self.labda_struct.items(), key=lambda x: x[1]))

		if self.averaged:
			print('averaging...')
			for l in self.labda_e:
				self.labda_e[l] = self.labda_e[l] - (self.labdaC_e[l] / C)
			for l in self.labda_ee:
				self.labda_ee[l] = self.labda_ee[l] - (self.labdaC_ee[l] / C)
			for s in self.labda_struct:
				self.labda_struct[s] = self.labda_struct[s] - (self.labdaC_struct[s] / C)
		#self.plot_loss_trajectory('loss.png')
				
			
	def predict(self, X, constraints = set(['MUL'])):
		Yp = []
		for i,X_k in enumerate(X):
			Yp.append(self.decode(X_k, constraints))
		return Yp
	
	def loss_both(self, Y_p,Y_k):
		return (self.loss(Y_p[0],Y_k[0]),self.loss(Y_p[1],Y_k[1]))

	def loss(self, Y_p,Y_k): # accuracy
		loss = 0
		len_y = len(Y_p)
		for i in range(len_y):
			loss += int(Y_p[i] != Y_k[i])
		return float(loss) / (len_y + 0.00001)
	
	def update(self,X_k, Y_k, Y_p, C, lr):
		Xe_k, Xee_k = X_k
		Ye_k,Yee_k = Y_k
		Ye_p,Yee_p = Y_p
		
		# Update Entity Weights:
		Phi_e_p = {l:csr_matrix((1,self.dims['e'])) for l in self.labels_e}
		Phi_e_k = {l:csr_matrix((1,self.dims['e'])) for l in self.labels_e}
		for i,obj in enumerate(Xe_k):
			Phi_e_p[Ye_p[i]] += obj.phi_v
			Phi_e_k[Ye_k[i]] += obj.phi_v
		for l in self.labda_e:
			diff = (Phi_e_k[l] - Phi_e_p[l]).toarray()[0]
			if self.balance:
				diff = diff * (1.0 / self.balance['e'][l])		
			if self.regularization =='l2':
				self.labda_e[l] = self.labda_e[l] * (1 - lr * self.regularization_term)

			self.labda_e[l] += lr * diff
			if self.averaged:
				if self.regularization =='l2':
					self.labdaC_e[l] = self.labdaC_e[l] * (1 - lr * self.regularization_term)
				self.labdaC_e[l] += C * lr * diff
				
		# Update TLink Weights:
		Phi_ee_p = {l:csr_matrix((1,self.dims['ee'])) for l in self.labels_ee}
		Phi_ee_k = {l:csr_matrix((1,self.dims['ee'])) for l in self.labels_ee}
		for i,obj in enumerate(Xee_k):
			Phi_ee_p[Yee_p[i]] += obj.phi_v 
			Phi_ee_k[Yee_k[i]] += obj.phi_v
		for l in self.labda_ee:
			diff = (Phi_ee_k[l] - Phi_ee_p[l]).toarray()[0]
			if self.balance:
				diff = diff  * (1.0 / self.balance['ee'][l])
			if self.regularization =='l2':
				self.labda_ee[l] = self.labda_ee[l] * (1 - lr * self.regularization_term)
				
			self.labda_ee[l] +=  lr * diff 		
			
			if self.averaged:
				if self.regularization =='l2':
					self.labdaC_ee[l] = self.labdaC_ee[l] * (1 - lr * self.regularization_term)
				self.labdaC_ee[l] += C * lr * diff

		# Update Structured Feature Weights
		Phi_struct_p = self.structured_feature_handler.extract(X_k, Y_p)
		Phi_struct_k = self.structured_feature_handler.extract(X_k, Y_k)
		
		for struct_feat in Phi_struct_k.keys():
			diff = Phi_struct_k[struct_feat] - Phi_struct_p[struct_feat]
			self.labda_struct[struct_feat] += lr * diff
			if self.averaged:
				self.labdaC_struct[struct_feat] +=  C * lr * diff
					
	def decode(self, X, constraints, gurobi_model_out = False, loss_augmentation=False, gurobi_model=None,j=None):
		"""Solving the prediction by defining it as an ILP in Gurobi"""
		
		X_e, X_ee = X
		if not gurobi_model:
			model = grb.Model('Temporal-' + str(j))
		else:
			model = gurobi_model	
			

		label_scores_e = {'Ie:' +obj.ID() +':' + label: (1.0 / self.balance['e'][label]) * obj.phi_v.dot(self.labda_e[label])[0] for label in self.labels_e for i,obj in enumerate(X_e)}
		label_scores_ee = {'Iee:' +obj.ID() +':' + label: (1.0 / self.balance['ee'][label]) * obj.phi_v.dot(self.labda_ee[label])[0] for label in self.labels_ee for i,obj in enumerate(X_ee)}

		
		if not gurobi_model:
			# making decision variables
			vars_e = {}
			vars_ee = {}		
			for Ie in label_scores_e:
				vars_e[Ie] = model.addVar(vtype=grb.GRB.BINARY, name=Ie) #,obj=label_scores_e[Ie],
			for Iee in label_scores_ee:
				vars_ee[Iee] = model.addVar(vtype=grb.GRB.BINARY, name=Iee) # obj=label_scores_ee[Iee], 


			if self.structured_feature_handler:
				if self.structured_feature_handler.TLINK_argument_bigrams:
					for a1 in set([tlink.get_e1().ID() for tlink in X_ee]):
						for rel in self.labda_ee:
								model.addVar(vtype=grb.GRB.BINARY, name=a1 + ':arg1:' + rel)
					for a2 in set([tlink.get_e2().ID() for tlink in X_ee]):
						for rel in self.labda_ee:
								model.addVar(vtype=grb.GRB.BINARY, name=a2 + ':arg2:' + rel)
								
					
				struct_feats = self.structured_feature_handler.extract(X).items()
				for struct_feat, feat_obj in struct_feats:
					for prod_expression in feat_obj:
						model.addVar(vtype=grb.GRB.BINARY, name='*'.join(prod_expression))
			model.update()
		
			# adding constraints		
			if 'MUL' in constraints: # mutually exclusive labels
				for i, obj in enumerate(X_e):
					model.addConstr(grb.quicksum(model.getVarByName('Ie:' + obj.ID() +':' + label) for label in self.labels_e) == 1, 'MULe_' + obj.ID())
				for i, obj in enumerate(X_ee):
					model.addConstr(grb.quicksum(model.getVarByName('Iee:' + obj.ID() +':' + label) for label in self.labels_ee) == 1, 'MULee_' + obj.ID())	
			
			if 'Ctrans' in constraints or 'Btrans' in constraints: # transitivity of containment, and temporal order
				for i, obj_1 in enumerate(X_ee):
					for j, obj_2 in enumerate(X_ee):
						if i!=j and obj_1.get_e2() == obj_2.get_e1():
							var_obj_3 = obj_1.get_e1().ID() + '-' + obj_2.get_e2().ID()
							
							#note:  2 - A - B + C >= 1 <<<<corresponds to>>> (A and B) implies C
							if 'Ctrans' in constraints and 'Iee:' + var_obj_3 + ':' + 'CONTAINS' in label_scores_ee:
								model.addConstr(2 - model.getVarByName("Iee:" + obj_1.ID() + ":CONTAINS") - model.getVarByName("Iee:" + obj_2.ID() + ":CONTAINS") + model.getVarByName("Iee:" + var_obj_3 + ":CONTAINS") >= 1)
							if 'Btrans' in constraints and 'Iee:' + var_obj_3 + ':' + 'BEFORE' in label_scores_ee:	
								model.addConstr(2 - model.getVarByName("Iee:" + obj_1.ID() + ":BEFORE") - model.getVarByName("Iee:" + obj_2.ID() + ":BEFORE") + model.getVarByName("Iee:" + var_obj_3 + ":BEFORE") >= 1)
			
			if len([c for c in constraints if c in set(['C_CBB','C_CAA','C_BBB','C_BAA'])]) > 0:
				for i,obj in enumerate(X_ee):
						if "Ie:" + obj.get_e1().ID() + ":BEFORE" in label_scores_e and "Ie:" + obj.get_e2().ID() + ":BEFORE" in label_scores_e:
							if 'C_CBB' in constraints:	
								# [C_CBB] (contains(X,Y) and before(X,doctime)) --> before(Y,doctime)
								model.addConstr(2 - model.getVarByName("Iee:" + obj.ID() + ":CONTAINS") - model.getVarByName("Ie:" + obj.get_e1().ID() + ":BEFORE") + model.getVarByName("Ie:" + obj.get_e2().ID() + ":BEFORE") >= 1)
							if 'C_CAA' in constraints:	
								# [C_CAA] (contains(X,Y) and after(X,doctime)) --> after(Y,doctime)
								model.addConstr(2 - model.getVarByName("Iee:" + obj.ID() + ":CONTAINS") - model.getVarByName("Ie:" + obj.get_e1().ID() + ":AFTER") + model.getVarByName("Ie:" + obj.get_e2().ID() + ":AFTER") >= 1)
							if 'C_BBB' in constraints:
								# [C_BBB] (before(X,Y) and before(Y,doctime)) --> before(X,doctime)
								model.addConstr(2 - model.getVarByName("Iee:" + obj.ID() + ":BEFORE") - model.getVarByName("Ie:" + obj.get_e2().ID() + ":BEFORE") + model.getVarByName("Ie:" + obj.get_e1().ID() + ":BEFORE") >= 1)
							if 'C_BAA' in constraints:								
								# [C_BAA] (before(X,Y) and after(X,doctime)) --> after(Y,doctime)
								model.addConstr(2 - model.getVarByName("Iee:" + obj.ID() + ":BEFORE") - model.getVarByName("Ie:" + obj.get_e1().ID() + ":AFTER") + model.getVarByName("Ie:" + obj.get_e1().ID() + ":AFTER") >= 1)


		if gurobi_model:
			struct_feats = self.structured_feature_handler.extract(X).items()
	
		obj_e =  grb.quicksum(label_scores_e[Ie]*model.getVarByName(Ie) for Ie in label_scores_e)
		obj_ee = grb.quicksum(label_scores_ee[Iee]*model.getVarByName(Iee) for Iee in label_scores_ee)
		obj = obj_ee + obj_e

		if self.structured_feature_handler:
			if self.structured_feature_handler.TLINK_argument_bigrams:
				
				# make sure that if REL(a,b) <--> arg1(a,REL) and arg2(b,REL)
				for i,tlink in enumerate(X_ee):
					for label  in self.labels_ee:
							model.addConstr(model.getVarByName("Iee:" + tlink.ID() + ':' + label) - model.getVarByName(tlink.get_e1().ID() + ':arg1:' + label) == 0)
							model.addConstr(model.getVarByName("Iee:" + tlink.ID() + ':' + label) - model.getVarByName(tlink.get_e2().ID() + ':arg2:' + label) == 0)
			
			
			for struct_feat, feat_obj in struct_feats:				
					for prod_expression in feat_obj:
						obj += self.labda_struct[struct_feat] * model.getVarByName('*'.join(prod_expression))
						model.addConstr(len(prod_expression) - grb.quicksum(model.getVarByName(expr) for expr in prod_expression) + model.getVarByName('*'.join(prod_expression)) >= 1)
						
		
		if loss_augmentation and self.loss_augmented_training: 
			Ye_aug, Yee_aug = loss_augmentation
			augmented_obj_e = grb.quicksum(model.getVarByName('Ie:' +e.ID() +':' + y_l)*label_scores_e['Ie:' +e.ID() +':' + y_l] for i,(e,y_l) in enumerate(zip(X_e,Ye_aug)))
			augmented_obj_ee = grb.quicksum(model.getVarByName('Iee:' +ee.ID() +':' + y_l)*label_scores_ee['Iee:' +ee.ID() +':' + y_l] for i,(ee,y_l) in enumerate(zip(X_ee,Yee_aug)))
						
			obj = obj - augmented_obj_e - augmented_obj_ee
			
			model.setObjective(obj,grb.GRB.MAXIMIZE)	
		else:
			model.setObjective(obj,grb.GRB.MAXIMIZE)	
		

		model.setParam( 'OutputFlag', False )
		model.update()
		model.params.optimalitytol = 1e-8
		model.params.timelimit = 10
		model.optimize()

		# interpreting the gurobi output
		out_e = {}
		out_ee = {}
		for v in model.getVars():
			if not (v.varName in vars_e or v.varName in vars_ee): # ignore the structured variables
				continue

			if v.x:
				varType, varID, varLabel = v.varName.split(':')
				if varType == 'Ie':
					out_e[varID] = varLabel
				elif varType == 'Iee':
					out_ee[varID] = varLabel
				
		Yp_e = [out_e[obj.ID()] for i,obj in enumerate(X_e)]
		Yp_ee = [out_ee[obj.ID()] for i,obj in enumerate(X_ee)]
		
		if gurobi_model_out:
			return ((Yp_e, Yp_ee),model)
		else:
			return (Yp_e, Yp_ee)	

	def get_negative_sample(self, X,Y,sample_size=10,typ='random'):
		(X_e, X_ee),(Y_e, Y_ee) = X,Y
		Xn_ee, Yn_ee = [],[]
		label_indices = {l:[] for l in self.labels_ee}
		for i,(x_ee,y_ee) in enumerate(zip(X_ee, Y_ee)):
			label_indices[y_ee].append(i)
		
		end =len(label_indices['no_label'])-sample_size
		random.shuffle(label_indices['no_label'])	
		for l in label_indices:
			if l!='no_label':
				for index in label_indices[l]:
					start =  random.randint(0,end)
					
					if typ=='random':
						neg_example_index = label_indices['no_label'][start]
					if typ=='loss_augmented':
						neg_example_index = max(label_indices['no_label'][start:start+sample_size], key=lambda neg,pos=index, Obs=X_ee,lab=self.labda_ee: Obs[neg].phi_v.dot(lab[l])[0])						
					
					Xn_ee.append(X_ee[neg_example_index])
					Yn_ee.append(Y_ee[neg_example_index])
					Xn_ee.append(X_ee[index])
					Yn_ee.append(Y_ee[index])
		return ((X_e, Xn_ee), (Y_e, Yn_ee))	
					
	def save_model(self, path):
		with open(path,'wb') as f:
			pickle.dump(self, f)

def load_sp_model(path):
	with open(path, 'rb') as f:
		SP = pickle.load(f)
	return SP



