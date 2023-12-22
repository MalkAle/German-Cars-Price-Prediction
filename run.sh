#!/bin/bash

#Get Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

#Clone repo
git clone https://github.com/MalkAle/German-Cars-Price-Prediction.git


sudo apt install python3-pip -y

#Temporary running
#python3 -m streamlit run app.py
#Permanent running
nohup python3 -m streamlit run ger_cars_app.py