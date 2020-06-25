# SPTempRels

SPTempRels trains and evaluates a structured perceptron model for extracting temporal relations from clinical texts, in which events and temporal expressions are given. It can also be used to replicate the experiments done by [Leeuwenberg and Moens (EACL, 2017)](http://www.aclweb.org/anthology/E/E17/E17-1108.pdf). The paper contains a detailed description of the model. The conference slides can be found [here](https://github.com/tuur/SPTempRels/raw/master/SPTempRels-EACL2017-Slides.pdf). When using this code please refer to the paper.


> Any questions? Feel free to send me an email at aleeuw15@umcutrecht.nl




## Reference
> In case of usage, please cite the corresponding publications.

```
@InProceedings{leeuwenberg2017structured:EACL,
  author    = {Leeuwenberg, Artuur and Moens, Marie-Francine},
  title     = {Structured Learning for Temporal Relation Extraction from Clinical Records},
  booktitle = {Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
}
```


### Requirements
* [Gurobi](https://www.gurobi.com) 
- create account, download gurobi, and run setup.py
* [Python2.7](https://www.python.org/downloads/release/python-2711/)
  * [Argparse](https://pypi.python.org/pypi/argparse)
  * [Numpy](http://www.numpy.org/)
  * [SciPy](https://www.scipy.org/)
  * [Networkx](https://networkx.github.io)
  * [Scikit-Learn](http://scikit-learn.org/stable/)
  * [Pandas](http://pandas.pydata.org/)


When cTAKES output is not provided the program backs off to the [Stanford POS tagger](http://nlp.stanford.edu/software/tagger.shtml) for POS features. For this reason it is required to have the Stanford POS Tagger folder (e.g. `stanford-postagger-2015-12-09`), the `stanford-postagger.jar`, and the `english-bidirectional-distsim.tagger` file at the same level as `main.py`.

### Data

In the paper we used the [THYME](https://clear.colorado.edu/TemporalWiki/index.php/Main_Page) corpus sections as used for the [Clinical TempEval 2016](http://alt.qcri.org/semeval2016/task12/index.php?id=data) shared task. So, training, development, or test data should be provided in the anafora xml format, in the folder structure as indicated below, where in the deepest level contains the text file `ID001_clinic_001` and corresponding xml annotations `ID001_clinic_001.Temporal-Relation.gold.completed`. Notice that we refer to the top level of the THYME data (`$THYME`) also in the python calls below.

`$THYME`
* `Train`
  * `ID001_clinic_001`
    * `ID001_clinic_001`     
    * `ID001_clinic_001.Temporal-Relation.gold.completed.xml`
  * ...
* `Dev`
  * ... 
* `Test`
  * ...

In our experiments we use POS, and dependency parse features from the [cTAKES Clincal Pipeline](http://ctakes.apache.org/). So, you need to provide the cTAKES output xml files as well. Here we assume these are in a directory called `$CTAKES_XML_FEATURES`. You can also call the program without the -ctakes_out argument. Then the it will use the Stanford POS Tagger for POS tag features instead (and no dependency parse features). The folder structure of this directory is:

`$CTAKES_XML_FEATURES`
* `ID001_clinic_001.xml`
* ...

### Experiments: Leeuwenberg and Moens (2017)
To obtain the predictions from the experiments of section 4 in the paper you can use the example calls below. Each call will output the anafora xmls to the directory `$SP_PREDICTIONS`. To get more information about the usage of the tool you can run:
```
python main.py -h
```

#### SP
```sh
python main.py $THYME 1 0 32 MUL 1000 Test -averaging 1 -local_initialization 1 -negative_subsampling 'loss_augmented' -lowercase 1 -lr 1 -output_xml_dir $SP_PREDICTIONS -constraint_setting CC -ctakes_out_dir $CTAKES_XML_FEATURES -decreasing_lr 0
```

#### SP random
```sh
python main.py $THYME 1 0 32 MUL 1000 Test -averaging 1 -local_initialization 1 -negative_subsampling 'random' -lowercase 1 -lr 1 -output_xml_dir $SP_PREDICTIONS -constraint_setting CC -ctakes_out_dir $CTAKES_XML_FEATURES -decreasing_lr 0
```

#### SP + 𝒞 *

```sh
python main.py $THYME 1 0 32 MUL,Ctrans,Btrans,C_CBB,C_CAA,C_BBB,C_BAA  1000 Test -averaging 1 -local_initialization 1 -negative_subsampling 'loss_augmented' -lowercase 1 -lr 1 -output_xml_dir $SP_PREDICTIONS -constraint_setting CC -ctakes_out_dir $CTAKES_XML_FEATURES -decreasing_lr 0
```


#### SP + 𝚽sdr
```sh
python main.py $THYME 1 0 32 MUL  1000 Test -averaging 1 -local_initialization 1 -negative_subsampling 'loss_augmented' -lowercase 1 -lr 1 -output_xml_dir $SP_PREDICTIONS -constraint_setting CC -ctakes_out_dir $CTAKES_XML_FEATURES -decreasing_lr 0 -structured_features DCTR_bigrams,DCTR_trigrams
```




