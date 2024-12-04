# Strategic TRansport Model (STraM)
Open source version of STraM in Pyomo

# Licencing
The STraM framework and all additional files in the git repository are licensed under the MIT license. In short, that means you can use and change the code of STraM. Furthermore, you can change the license in your redistribution but must mention the original author. We appreciate if you inform us about changes and send a merge request via git. For further information please read the LICENSE file, which contains the license text, or go to https://opensource.org/licenses/MIT

# Required Software
STraM is available in the Python-based, open-source optimization modelling language Pyomo. Running the model thus requires some coding skills in Python. To run the model, make sure Python, Pyomo and a third-party solver (e.g., gurobi, or CPLEX) is installed and loaded to the respective computer. More information on how to install Python and Pyomo can be found here: http://www.pyomo.org/installation. 
Other python package dependencies can be found in the file requirements.txt.

To download, you need to install Git and clone the repository. 
## To activate venv (for Windows):
Before starting, make sure you have installed and updated pip

1) Download Python 3.10.4 
2) Create virtual environment: py -3.10 -m venv env
3) Activate virutal environment: .\env\Scripts\activate
4) Install packages: pip install -r requirements.txt
5) Verify installations: python -m pip list

## To activate venv (for Mac or linux):
Before starting, make sure you have installed and updated pip

1) Download Python 3.10.4 
2) Create virtual environment: python3.10 -m venv env
3) Activate virutal environment: source env/bin/activate
4) Install packages: pip install -r requirements.txt
5) Verify installations: python -m pip list

# Software Structure
STraM consists of various programming scripts, including: 

<b>(1)	Main.py:</b> The main script used to run STraM. It links to all other scripts. This is the main script a user needs to use and potentially modify. There are various analyses that can be selected. 

<b>(2)	Model.py:</b> Contains the model formulation of STraM in Pyomo. 

<b>(3)	Data/ConstructData.py:</b> this file constructs the necessary data structure based on all excel files.


