To generate virtual species data, enter the code folder and run the files in the following order:

1_sim_landscape.py
2_sim_settlement.py
3_sim_export.py

The first file generates a virtual environment for the virtual species (with the class landscape from landscape.py).
The second file generates the distribution of the virtual species (with the class settlement from settlement.py).
The third file converts the distribution to table-like data and saves it to a txt file (with the class export from export.py).

The scripts rely on:
scenarios.json and settings.json: Files where the settings of the simulations are stored. These are shared between the scripts.
code/functionalities.py: This is a file with easy functions that are shared between the scripts.

In the seeds folder, we stored all the seeds that were generated in the generation of the simulations. To reproduce our results, these have to be used instead of the usual generation of new random seeds.
