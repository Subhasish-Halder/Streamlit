Development Setup Guide
This project uses Python virtual environments to ensure consistent dependencies across all developers' machines. Follow the steps below to set up your environment.
1️. Update and Upgrade System Packages
•	Run the following command:
sudo apt update && sudo apt upgrade -y
2️. Install Python and Virtual Environment Tools
•	Run the following command:
sudo apt install python3 python3-venv python3-pip -y
3️. Create a Virtual Environment
•	Inside the project directory, run:
python3 -m venv st
4️. Activate the Virtual Environment
•	Run:
source st/bin/activate
When active, your terminal prompt will show (st) at the beginning.
5️. Upgrade Core Tools
•	Run:
pip install --upgrade pip setuptools wheel
6️. Install Project Dependencies
•	Run:
pip install -r requirements.txt
7️. Deactivate the Virtual Environment
•	Run:
deactivate

Workflow Summary
•	Activate the environment before running scripts:
source st/bin/activate
•	Install new dependencies:
pip install <package>
•	Update requirements.txt after installing new packages:
pip freeze > requirements.txt
•	Deactivate when finished:
deactivate
 With this setup, every developer can work in an isolated environment without breaking system-wide Python packages.
