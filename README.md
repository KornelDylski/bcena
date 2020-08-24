# BCENa loss function

Notebook contains experiments for BCENa loss, which is tweaked BCE loss that takes into account unknown and unseen before categories. Checks whether changing the target vector of an unknown category to an empty vector would be more effective (or less) than treating it the same as the others.

Code is written in pytorch with fastai2<br>
Dataset used for experiments is Imagenett

**labs/bcena-experiments.ipynb** - notebook containing experiment

**bcena-loss** - python module with BCENa loss and customized metrics