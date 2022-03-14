#!bin/bash
echo "start server"
cd /home/ec2-user/yuddomack
/bin/echo "Hello World" >> testfile1.txt
python manage.py runserver 0:8000
python ex.py
/bin/echo "Hello World" >> testfile2.txt
