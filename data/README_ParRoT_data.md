# The ParRoT Dataset

The ParRoT dataset is designed to test whether language models have coherent mental models of everyday things (e.g. egg, tree, flashlight).
The dataset is described in "Do language models have coherent mental models of everyday things?" 
We treat the task as that of constructing a "parts mental model" for everyday things, and evaluate if language models can accurately judge whether each relation statement is True/False.
The dataset contains 11720 relations (p1, rln, p2) across 100 everyday things, 300 mental models, and 2191 parts.

Relationships encoded include:
* spatial orientation (part of, has part, inside, contains, in front of, behind, above, below, surrounds, surrounded by, next to)
* connectivity (connects) \*A connects B denotes to is directly connected to B
* functional dependency (requires, required by) \*A requires B denotes A cannot perform its primary function without B
  
We have also included the ids indicating different workers that sketched the parts mental models.

We make our dataset publicly avilable here: [Dropbox folder for ParRoT-release](https://www.dropbox.com/sh/tv2hc6pmsbr25l3/AAAXZKvfkfyx6SAkqjolhS0ra?dl=0).

This release includes the following:
The full-triplets-2022Dec16-11720-release.tsv contains all the transcribed relations in the form of (p1, rln, p2) triplets.
The ParRoT_MM_sketches.zip is a compressed folder that contains the 300 parts mental models sketched by crowd workers, in .png format. 
