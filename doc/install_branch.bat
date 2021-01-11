git clone -b x7_refactor https://gribbg@bitbucket.org/dubnom/gears.git gears_branch
cd gears_branch
"c:\Program Files\Python38"\python -m venv venv
call .\venv\Scripts\activate.bat
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m view.main
