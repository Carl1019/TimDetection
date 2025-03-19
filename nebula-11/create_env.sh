echo "Creating new virtualenv"

python3 -m venv ~/$1
source ~/$1/bin/activate


echo "Installing Requirements"

pip install -r requirements.txt