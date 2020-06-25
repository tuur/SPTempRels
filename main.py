from __future__ import print_function, division
from structured_perceptron import StructuredPerceptron, load_sp_model
from thyme import read_thyme,write_to_anafora
from features import DocTimeRelFeatureHandler, TLinkFeatureHandler, StructuredFeatureHandler
import numpy as np
from sklearn import linear_model
from evaluation import Evaluation
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import vstack
import argparse
import sys
import pickle as pickle
sys.setrecursionlimit(50000)

parser = argparse.ArgumentParser(description='Implementation for a Structured Perceptron for Temporal Relation Extraction.')
parser.add_argument('thyme', type=str,
                    help='thyme corpus folder with subdirectories Train, Dev, and Test')
parser.add_argument('-sp', type=int,default=1,
                    help='run the structured perceptron')
parser.add_argument('-p', type=int,default=0,
                    help='run the the normal perceptron')																				
parser.add_argument('-it', type=int, default=32,
                    help='number of SP iterations')
parser.add_argument('-constraints', type=str, default = 'MUL',
                    help='constraints, separated by a comma (e.g. MUL, C-DCT, B-DCT, C-trans, B-trans) default: MUL')
parser.add_argument('-docs', type=int,default=1000000,
                    help='maximum number of documents used in each set (train, dev)')
parser.add_argument('-setting', type=str, default='Dev',
                    help='Dev of Test setting (default=Dev)')
parser.add_argument('-output_xml_dir', type=str, default=None,
                    help='Write to anafora xml output')
parser.add_argument('-la', type=int, default=0,
                    help='Use Loss Augmented Training (default=0)')
parser.add_argument('-regularization', type=str, default=None,
                    help='Use regularization for global learning: default:l2')
parser.add_argument('-regularization_term', type=float, default=1.0,
                    help='Specify the regularization term (when using L2 regularization), default:1.0')																						
parser.add_argument('-local_initialization', type=int, default=1,
                    help='Use local initialization for the structured perceptron (default:0)')	
parser.add_argument('-averaging', type=int, default=1,
                    help='Use averaging (default:1)')																				
parser.add_argument('-lr', type=float, default=1,
                    help='Learning rate for the Structured Perceptron (default:1)')	
parser.add_argument('-decreasing_lr', type=int, default=0,
                    help='Using a linearly decreasing learning rate (default:0)')	
parser.add_argument('-negative_subsampling', type=str, default=None,
                    help='Type of negative sub-sampling used:random or loss_augmented (default:loss_augmented)')
parser.add_argument('-constraint_setting', type=str, default='CC',
                    help='Use constraints during training and prediction (CC), or only during training (CU), or only during prediction (UC), or use no constraints (UU). default:CC')
parser.add_argument('-shuffle', type=int, default=1,
                    help='Shuffle during training (default=1)')		
parser.add_argument('-pit', type=int,default=8,
                    help='number of P iterations (default=8)')
parser.add_argument('-ctakes_out_dir', type=str,
                    help='Use cTAKES output features (POS and/or dependency path)')																					
parser.add_argument('-load_token_clusters', type=str, default=None,
                    help='Loads a cluster file of ngrams, and uses them as features.')																																										
parser.add_argument('-save_features', type=str, default=None,
                    help='Saves input features and candidates to a file.')	
parser.add_argument('-max_token_distance', type=int, default=30,
                    help='Maximum token distance between candidates (for candidate selection, default:30)')	
parser.add_argument('-tlinks', type=str, default='CONTAINS,BEFORE,OVERLAP,BEGINS-ON,ENDS-ON',
                    help='TLINK labels to predict (separated by a comma) default:CONTAINS,BEFORE,OVERLAP,BEGINS-ON,END-ON')	
parser.add_argument('-load_features', type=str, default=None,
                    help='Loads input features and candidates from a file.')
parser.add_argument('-save_document_structure', type=str, default=None,
                    help='Saves input document structure (with tokenization and POS).')
parser.add_argument('-load_document_structure', type=str, default=None,
                    help='Loads input document structure (with tokenization and POS).')
parser.add_argument('-structured_features', type=str, default='',
                    help='Comma separated structured feature specification: from DCTR_bigrams,DCTR_of_TLINKS,TLINK_argument_bigrams (default=None)')
parser.add_argument('-lowercase', type=int, default=1,
                    help='Consider all text as lowercased. default:1')
parser.add_argument('-conflate_digits', type=int, default=0,
                    help='Conflates all digits. default:0')
parser.add_argument('-execute', type=str, default=None,
                    help='Execute a part of code (e.g. to set features)')		
parser.add_argument('-pos', type=int, default=1,
                    help='Using POS features (default=1)')		
parser.add_argument('-save_model', type=str, default='model.p',
                    help='Saves model to file, default:model.p')
parser.add_argument('-load_model', type=str, default=None,
                    help='Loads SP model, default:None')
parser.add_argument('-set_pos_model', type=str, default='english-bidirectional-distsim.tagger',
                    help='Sets stanford POS Tagging model, default:english-bidirectional-distsim.tagger')
parser.add_argument('-tasks', type=str, default='DCTR,TLINK',
                    help='Tasks to be learned (default=DCTR,TLINK)')		
args = parser.parse_args()


# GENERAL HYPERPARAMETERS
structured_prediction = bool(args.sp)
local_prediction = bool(args.p)
labels_e = set(['BEFORE','AFTER','BEFORE/OVERLAP','OVERLAP'])
labels_ee = set(args.tlinks.split(',') + ['no_label'])
closure = ['CONTAINS','BEFORE']

print('DOCTIMEREL labels:',labels_e)
print('TLINK labels:',labels_ee)

max_token_distance = args.max_token_distance
normalise_features = False
same_par = True
max_num_documents = args.docs
feature_extractor_e = DocTimeRelFeatureHandler()
feature_extractor_ee = TLinkFeatureHandler()
feature_extractor_struct = StructuredFeatureHandler(labels_e, labels_ee)

for struct_f in args.structured_features.replace(' ','').split(','):
	if struct_f == 	'DCTR_bigrams':
		feature_extractor_struct.DCTR_bigrams = True
	if struct_f == 'DCTR_trigrams':
		feature_extractor_struct.DCTR_trigrams = True
	if struct_f == 	'DCTR_of_TLINKS':
		feature_extractor_struct.DCTR_of_TLINKS = True
	if struct_f == 	'TLINK_argument_bigrams':
		feature_extractor_struct.TLINK_argument_bigrams = True
		

thyme = args.thyme 
if args.load_document_structure:
	with open(args.load_document_structure, 'rb') as f:
		thyme_document_structure = pickle.load(f)
else:
	datasets = ['Train','Dev','Test']
	if args.load_model:
		datasets.remove('Train')
	thyme_document_structure = read_thyme(thyme,regex='.*Temp.*', max_documents=max_num_documents, closure=closure, lowercase=bool(args.lowercase), conflate_digits=bool(args.conflate_digits),ctakes_out_dir=args.ctakes_out_dir, pos=args.pos, pos_model=args.set_pos_model,datasets=datasets)
	if args.save_document_structure:
		with open(args.save_document_structure, 'wb') as f:
			pickle.dump(thyme_document_structure, f)


if not args.pos:
	print('no POS!')
	
	feature_extractor_e.entity_pos_tags = False
	feature_extractor_e.left_context_pos_bigrams = False
	feature_extractor_e.right_context_pos_bigrams = False
	feature_extractor_e.closest_verb = False
	feature_extractor_e.surrounding_entities_pos = False
	
	feature_extractor_ee.entity_pos_window = False
	feature_extractor_ee.entity_pos_tags = False
	feature_extractor_ee.pos_ngrams_inbetween = False


if args.load_token_clusters:
	for clustering in args.load_token_clusters.split(','):
		feature_extractor_e.read_token_clusters(clustering)
		feature_extractor_ee.read_token_clusters(clustering)
		feature_extractor_e.left_context_cl_bigrams, feature_extractor_e.right_context_cl_bigrams, feature_extractor_e.entity_tok_clusters =[3,5], [3,5], True
		feature_extractor_ee.entity_cls, feature_extractor_ee.cl_ngrams_inbetween, feature_extractor_ee.entity_cl_window  = True, True, 3

		
if args.setting == 'Test':
	thyme_document_structure['Train'] += thyme_document_structure['Dev']
	print('Training on Train+Dev:', len(thyme_document_structure['Train']),'documents')
		
		
		
if args.execute:
	exec args.execute
		
# GLOBAL HYPERPARAMETERS
negative_subsampling=args.negative_subsampling
plot_loss = False
averaged = bool(args.averaging)
balance = False
num_iterations = args.it
lr = args.lr
shuffle_training_data = bool(args.shuffle)
decreasing_lr = bool(args.decreasing_lr)
loss_augmented_training = bool(args.la)
local_initialization = bool(args.local_initialization)
regularization= args.regularization
regularization_term = args.regularization_term

constraints = set(args.constraints.split(',')) 
training_constraints, test_constraints = set(['MUL']), set(['MUL'])
if args.constraint_setting[0] == 'C':
	training_constraints = constraints
if args.constraint_setting[1] == 'C':
	test_constraints = constraints
print('training constraints:', training_constraints, 'testing constraints:', test_constraints)	

# LOCAL HYPERPARAMETERS


if args.load_features and not args.load_model:
	print('loading features from',args.load_features + '.train.p')
	with open(args.load_features + '.train.p','rb') as f:
		(X,Y,feature_extractor_e, feature_extractor_ee) = pickle.load(f)
elif not args.load_model:
	# ----------------------- Constructing Training Set
	X, Y = [], []
	for i,doc in enumerate(thyme_document_structure['Train']):
		
		print(i,'creating candidates for',doc.ID())
		X_e = doc.get_doctimerel_candidates()
		X_ee = doc.get_tlink_candidates(labels_ee, max_token_distance=max_token_distance,same_par=same_par)
		
		if not 'DCTR' in args.tasks.split(','):
			X_e = []
		if not 'TLINK' in args.tasks.split(','):
			X_ee = []		
		
		print('extracting features for ',doc.ID(),'e:',len(X_e),'ee:',len(X_ee))
		for e in X_e:
			e.phi = feature_extractor_e.extract(e, doc)

		for ee in X_ee:
			ee.phi = feature_extractor_ee.extract(ee, doc)
				
		
					
		Y_e = [e.get_doctimerel() for e in X_e]
		Y_ee = [tlink.get_tlink() for tlink in X_ee]


		
		doc.phi_struct = feature_extractor_struct.extract((X_e, X_ee),(Y_e, Y_ee))
		X.append((X_e, X_ee))		
		Y.append((Y_e, Y_ee))
				
	feature_extractor_e.update_vectorizer()
	feature_extractor_ee.update_vectorizer()
	feature_extractor_struct.update_vectorizer()
	
	print('vectorizing features')
	for i,doc in enumerate(X):
		for j,obj in enumerate(X[i][0]):
			obj.phi_v = feature_extractor_e.vectorize(obj.phi)
		for j,obj in enumerate(X[i][1]):
			obj.phi_v = feature_extractor_ee.vectorize(obj.phi)
		

	if args.save_features:
		print('saving training features to file...')
		with open(args.save_features + '.train.p', 'wb') as f:
			pickle.dump((X,Y,feature_extractor_e, feature_extractor_ee), f)

if not args.load_model:
	print('DocTimeRel Features:', len(feature_extractor_e.feature_template))
	print('TLink Features:', len(feature_extractor_ee.feature_template))


	# setting up data normalisation
	if normalise_features and not args.load_model:
		print('normalizing features')
		scaler_ee, scaler_e = preprocessing.MaxAbsScaler(), preprocessing.MaxAbsScaler()
		scaler_ee.fit(vstack([ee.phi_v for x_e,x_ee in X for ee in x_ee],format='csr'))
		scaler_e.fit(vstack([e.phi_v for x_e,x_ee in X for e in x_e], format='csr'))


	if local_prediction or local_initialization:
		print('== LOCAL TRAINING ==\n...')
	
		classifier_e = OneVsRestClassifier(linear_model.Perceptron(n_iter=args.pit))
		classifier_ee = OneVsRestClassifier(linear_model.Perceptron(n_iter=args.pit))
	
		local_label_encoder_ee = preprocessing.LabelEncoder()
		local_label_encoder_e = preprocessing.LabelEncoder()
		if 'DCTR' in args.tasks.split(','):
			classifier_e.fit(feature_extractor_e.vectorize([e.phi for x_e,x_ee in X for e in x_e]),local_label_encoder_e.fit_transform([e for y_e,y_ee in Y for e in y_e]))
		if 'TLINK' in args.tasks.split(','):
			classifier_ee.fit(feature_extractor_ee.vectorize([ee.phi for x_e,x_ee in X for ee in x_ee]),local_label_encoder_ee.fit_transform([ee for y_e,y_ee in Y for ee in y_ee]))
		print(classifier_ee)

if structured_prediction and not args.load_model:
	
	print('== GLOBAL TRAINING ==')
	labels = (labels_e,labels_ee)
	dims = (feature_extractor_e.dim, feature_extractor_ee.dim, feature_extractor_struct.dim)
	SP = StructuredPerceptron(labels, dims,averaged=averaged,loss_augmented_training=loss_augmented_training,balance=balance,structured_feature_handler=feature_extractor_struct, regularization=regularization, regularization_term=regularization_term, feature_extractor_e=feature_extractor_e, feature_extractor_ee=feature_extractor_ee)
	
	
	if local_initialization:
		labda_e, labda_ee = {l:np.ones(dims[0]) for l in labels_e},{l:np.ones(dims[1]) for l in labels_ee}
		
		if 'DCTR' in args.tasks.split(','):
			for label, weights in zip(local_label_encoder_e.classes_, classifier_e.coef_):
				labda_e[label] = weights

		if 'TLINK' in args.tasks.split(','):
			for label, weights in zip(local_label_encoder_ee.classes_, classifier_ee.coef_):
				print('setting',label,weights)
				labda_ee[label] = weights	

		SP.set_labda(labda_e,labda_ee)	
		

	SP.train(X,Y,num_iterations=num_iterations, constraints=training_constraints,learning_rate=lr,decreasing_lr=decreasing_lr,shuffle=shuffle_training_data,negative_subsampling=negative_subsampling)
	if args.save_model:
		print('saving model to',args.save_model)
		SP.save_model(args.save_model)

	if plot_loss:
		SP.plot_loss_trajectory()
	
	if 'TLINK' in args.tasks.split(','):

		weights_Cee = feature_extractor_ee.inverse_transform([SP.labda_ee['CONTAINS']])[0]
		weights_Nee = feature_extractor_ee.inverse_transform([SP.labda_ee['no_label']])[0]
		print('IMPORTANT TLINK:CONTAINS FEATURES')
		for (k,vc,vn) in sorted([(k,vc,weights_Nee[k]) for (k,vc) in weights_Cee.items() if k in weights_Nee], key=lambda x:abs(x[1]-x[2]), reverse=True)[:30]:
			print(vc,'\t',vn,'\t',k)	
	
	if 'DCTR' in args.tasks.split(',')	:
		weights_Oe = feature_extractor_e.inverse_transform([SP.labda_e['OVERLAP']])[0]
		weights_Be = feature_extractor_e.inverse_transform([SP.labda_e['BEFORE']])[0]
	
		print('IMPORTANT DCTR:OVERLAPvsBEFORE FEATURES')
		for (k,vc,vn) in sorted([(k,vc,weights_Oe[k]) for (k,vc) in weights_Be.items() if k in weights_Oe], key=lambda x:abs(x[1]-x[2]), reverse=True)[:30]:
			print(vc,'\t',vn,'\t',k)

	
	
if args.load_model:
	SP = load_sp_model(args.load_model)
	feature_extractor_e = SP.feature_extractor_e
	feature_extractor_ee = SP.feature_extractor_ee

if args.load_features:
	print('loading features from',args.load_features + '.' + args.setting + '.p')
	with open(args.load_features + '.' + args.setting + '.p','rb') as f:
		(X,Y) = pickle.load(f)
else:
	# ----------------------- Constructing Dev Set	
	X, Y = [], []
	for i,doc in enumerate(thyme_document_structure[args.setting]):
		
		print('creating candidates for',doc.ID())
		X_e = doc.get_doctimerel_candidates()
		X_ee = doc.get_tlink_candidates(labels_ee, max_token_distance=max_token_distance, same_par=same_par)
	
	
		if not 'DCTR' in args.tasks.split(','):
			X_e = []
		if not 'TLINK' in args.tasks.split(','):
			X_ee = []
	
		print('extracting features for ',doc.ID())
		for e in X_e:
			e.phi = feature_extractor_e.extract(e, doc, update=False)
	
		for ee in X_ee:
			ee.phi = feature_extractor_ee.extract(ee, doc, update=False)



	
		X.append((X_e, X_ee))
		
		Y_e = [e.get_doctimerel() for e in X_e]
		Y_ee = [tlink.get_tlink() for tlink in X_ee]	
		Y.append((Y_e, Y_ee))

	print('vectorizing features')
	for i,doc in enumerate(X):
		for j,obj in enumerate(X[i][0]):
			obj.phi_v = feature_extractor_e.vectorize(obj.phi)
		for j,obj in enumerate(X[i][1]):
			obj.phi_v = feature_extractor_ee.vectorize(obj.phi)

	if args.save_features:
		print('saving',args.setting,'features to file...')
		with open(args.save_features + '.' + args.setting + '.p', 'wb') as f:
			pickle.dump((X,Y), f)

if structured_prediction:
	print('predicting...')
	Ys_p = SP.predict(X, test_constraints)
	print('Dev loss_e:',np.mean([SP.loss_both(Ys_p[i], Y[i])[0] for i in range(len(Y))]),'Dev loss_ee:',np.mean([SP.loss_both(Ys_p[i], Y[i])[1] for i in range(len(Y))]))
	evaluation = Evaluation(Ys_p, Y, name='GLOBAL LEARNING', tasks=args.tasks)
	
if (local_prediction or local_initialization) and not args.load_model:
	Yl_p = []
	for xe,xee in X:
		if 'TLINK' in args.tasks.split(','):
			Yl_p_ee = local_label_encoder_ee.inverse_transform(classifier_ee.predict(feature_extractor_ee.vectorize([ee.phi for ee in xee])))
		else:
			Yl_p_ee = []
		
		if 'DCTR' in args.tasks.split(','):	
			Yl_p_e = local_label_encoder_e.inverse_transform(classifier_e.predict(feature_extractor_e.vectorize([e.phi for e in xe])))
		else:
			Yl_p_e = []
		Yl_p.append((Yl_p_e,Yl_p_ee))
	evaluation = Evaluation(Yl_p, Y, name='LOCAL LEARNING', tasks=args.tasks)

if args.output_xml_dir:
	print('writing to anafora...')
	preds = []
	if local_prediction:
		preds.append((Yl_p,'local'))
	if structured_prediction:
		preds.append((Ys_p,'global'))
	write_to_anafora(X, preds, args.output_xml_dir, thyme_document_structure[args.setting])	
		
		

