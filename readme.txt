##########  Feb 12, 2025
Two python scripts to perform LSTM training and prediction: z.s1.train.py & z.s2.predict.17-20.py

A configure file is needed to specify inputs, output directory and other parameters. The file "main.config.test.23.crossvd" is provided as an example. Correspoding sample datasets are also included in "./data/" folder.

Use z.s1.train.py for training and save model (with pytorch).

Use z.s2.predict.17-20.py to generate csv outputs. The section of converting data to monthly scale can be removed if monthly records are not needed. 

