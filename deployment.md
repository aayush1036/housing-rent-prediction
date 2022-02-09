# Steps for deployment of the web app 

1. Create a non root user and give admin priviliges
```
# create a user
adduser YOUR_USER_NAME
```

# elevate the user to sudo group 
```
usermod -aG sudo YOUR_USER_NAME
``` 
2. Configure the firewall
```
# Create inbound rules 
ufw allow OpenSSH 
ufw allow http
ufw allow https 
# Enable the firewall
ufw enable 
```

3. Login from the newly created user 

4. Edit the sysctl.conf file to track large changes 
```
sudo vim /etc/sysctl.conf
```
# Paste this 
```
fs.inotify.max_user_watches=524288
```
# execute this command to set the value 
```
sudo sysctl -p 
```

5. Update package list and upgrade packages installed on this machine 
```
sudo apt update && sudo apt upgrade 
```

6. Install python and nginx 
```
sudo apt install python3-pip python3-dev nginx 
```
7. Fix the path (for avoiding the path warning while instaling packages)
```
export PATH=/Users/YOUR_USER_NAME/Library/Python/3.8/bin:$PATH
```
8. Configure github and clone the repository for getting the code in the server
```
# Configure  github 
git config --global user.name "YOUR_GITHUB_USER_NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```
# clone the repo from github 
```
git clone REPOSITORY_URL
```
9. Move to that directory
```
cd housing-rent-prediction
```
10. Create config.json and store the database details in it as it was not cloned because of gitignore file 

11. Create an environment and configure it
```
# Install virtualenv package to create environment named env 
sudo pip3 install virtualenv
```
# Create the environment
```
virtualenv env 
```
# activate the environment
```
source env/bin/activate 
```
# Install the required packages
```
pip3 install -r requirements.txt && pip3 install gunicorn 
```
12. Create ```wsgi.py``` as entry point 
```python
from main import app

if __name__ == '__main__':
    app.run()
```
13. Bind gunicorn to the app so that it can fetch the app from here 
```
gunicorn --bind 0.0.0.0:5000 wsgi:app
```
14. Deactivate the environment
```
deactivate  
```

15. Create and configure a service named app
```
sudo vim /etc/systemd/system/app.service
```
And paste the following in that file
``` 
[Unit]
#  specifies metadata and dependencies
Description=Gunicorn instance to serve myproject
After=network.target
# tells the init system to only start this after the networking target has been reached
# We will give our regular user account ownership of the process since it owns all of the relevant files
[Service]
# Service specify the user and group under which our process will run.
User=YOUR_USER_NAME
# give group ownership to the www-data group so that Nginx can communicate easily with the Gunicorn processes.
Group=www-data
# We'll then map out the working directory and set the PATH environmental variable so that the init system knows where our the executables for the process are located (within our virtual environment).
WorkingDirectory=/home/YOUR_USER_NAME/housing-rent-prediction/
#paste the path to bin folder of env
Environment="PATH=/home/YOUR_USER_NAME/housing-rent-prediction/env/bin" 
# We'll then specify the commanded to start the service
ExecStart=/home/YOUR_USER_NAME/housing-rent-prediction/env/bin/gunicorn --workers 3 --bind unix:app.sock -m 007 wsgi:app
# This will tell systemd what to link this service to if we enable it to start at boot. We want this service to start when the regular multi-user system is up and running:
[Install]
WantedBy=multi-user.target
```
Enable and start the service named app
```
sudo systemctl start app
sudo systemctl enable app
```
16. Configure nginx
Open the nginx file 
```
sudo vim /etc/nginx/sites-available/app
```
# Open in insert mode and paste the following thing 
```
server {
listen 80;
server_name YOUR_DOMAIN_NAME;

location / {
  include proxy_params;
  proxy_pass http://unix:/home/YOUR_USER_NAME/housing-rent-prediction/app.sock;
    }
location /static {
    alias /home/YOUR_USER_NAME/housing-rent-prediction/static;
}
location /Objects {
    alias /home/YOUR_USER_NAME/housing-rent-prediction/Objects;
}
location /templates {
    alias /home/YOUR_USER_NAME/housing-rent-prediction/templates;
}
}   
```
# Activate nginx configuration 
```
sudo ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled
```
# Restart nginx service 
```
sudo systemctl restart nginx
```
# Allow nginx through firewall
```
sudo ufw allow 'Nginx Full'
ufw enable 
```
17. Get SSL Certificate from certbot 
```
# Install and refresh snapd on your server
sudo snap install core; sudo snap refresh core
# Remove previous installations of certbot (if any)
sudo apt-get remove certbot
# Install certbot
sudo snap install --classic certbot
# Prepare the certbot command to run in your machine 
sudo ln -s /snap/bin/certbot /usr/bin/certbot
# Allow certbot to edit the nginx configuration files automatically 
sudo certbot --nginx
# create cron job to retrain model and renew SSL certificates every month 
crontab -e # edit crontab in vim editor 
# switch to insert mode and type the following commands 
# schedule a cron job to renew SSL certificates at 4:30 AM on 1st of every month
30 4 1 * * sudo certbot renew --quiet
```
18. Create a cron job which retrains the machine learning model 
Get path of python3 in env 
```
source env/bin/activate 
whereis python3 
```
Get the path of git 
```
whereis git 
```
Open crontab
```
crontab -e
```

Copy the path which has housing-rent-prediction in it <br>
Enable cron job for retraining model monthly at 12:00 AM


```
0 0 1 * * cd housing-rent-prediction && /home/YOUR_USER_NAME/housing-rent-prediction/env/bin/python3 combine.py && sudo service app restart && sudo service nginx restart && /usr/bin/git add --a && /usr/bin/git commit -m "Run cron job" && /usr/bin/git push origin master >> outputs.txt
```