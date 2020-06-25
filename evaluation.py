from __future__ import print_function, division
from collections import Counter
import pandas as pd

class Evaluation:

	def __init__(self, Y_p, Y, name='', tasks='DCTR,TLINK'):
		self.name = name
		self.Y_p, self.Y = Y_p, Y
		print('\n---> EVALUATION:',self.name,'<---')
		if 'DCTR' in tasks.split(','):
			self.evaluate_e()
		if 'TLINK' in tasks.split(','):
			self.evaluate_ee()
		
	def pprint(self):
		return 'todo'
		
	def evaluate_e(self):
		print('\n*** Evaluating DOCTIMEREL ***')
		self.evaluate([yp[0] for yp in self.Y_p], [y[0] for y in self.Y])
	
	def evaluate_ee(self):
		print('\n*** Evaluating TLINKS ***')
		self.evaluate([yp[1] for yp in self.Y_p], [y[1] for y in self.Y])

	def evaluate(self, Yp, Y): # Internal evaluation, may not be the same as in Clinical TempEval (due to temporal closure and candidate generation)!
		Yp = [l for i in Yp for l in i]
		Y = [l for i in Y for l in i]
		labels = set(Y+Yp)
		print('Y:',set(Y),'Yp',set(Yp))
		y_actu = pd.Series(Y, name='Actual')
		y_pred = pd.Series(Yp, name='Predicted')
		confusion = Counter(zip(Y,Yp))
		df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
		print('==CONFUSION MATRIX==')
		print(df_confusion)
		print('==PER LABEL EVALUATION==')
		print('  P\t   R\t   F\t')
		s_TP, s_FP, s_FN = 0,0,0
		for l in labels:
			TP = confusion[(l,l)] if (l,l) in confusion else 0
			FP = sum([confusion[(i,l)] for i in labels if (i,l) in confusion and l!=i])
			FN = sum([confusion[(l,i)] for i in labels if (l,i) in confusion and l!=i])
			print('TP',TP,'FP',FP,'FN',FN)
			precision = float(TP) / (TP + FP + 0.000001)
			recall = float(TP) / (TP + FN + 0.000001)
			fmeasure = (2 * precision * recall) / (precision + recall + 0.000001)
			print(round(precision,4),'\t',round(recall,4),'\t',round(fmeasure,4),'\t',l)
			s_TP += TP
			s_FP += FP
			s_FN += FN
		s_prec = float(s_TP) / (s_TP + s_FP + 0.000001)
		s_recall = float(s_TP) / (s_TP + s_FN + 0.000001)
		s_fmeasure = (2 * s_prec * s_recall) / (s_prec + s_recall + 0.000001)
		print(round(s_prec,4),'\t',round(s_recall,4),'\t',round(s_fmeasure,4),'\t','**ALL**')



				
