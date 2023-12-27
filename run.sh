#!/bin/bash

sudo apt-get install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker $(whoami)
newgrp docker



#Temporary running
#python3 -m streamlit run app.py
#Permanent running
#nohup python3 -m streamlit run ger_cars_app.py