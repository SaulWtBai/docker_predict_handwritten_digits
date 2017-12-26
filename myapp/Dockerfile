FROM ubuntu

# 安装python 和 pip
RUN apt update \
        && apt install -y python3-pip python3-dev vim \
        && pip3 install --upgrade pip \
        && pip3 install tensorflow \
        && ln -s /usr/bin/python3 /usr/bin/python \
        && ln -s /usr/bin/pip3 /usr/bin/pip \
        && pip3 install Flask==0.10.1 \
        && pip3 install Pillow \
        && pip3 install cassandra-driver

# copy files required for the app to run
COPY app.py /Users/wentanbai/Desktop/code/myapp/
COPY templates/upload.html /Users/wentanbai/Desktop/code/myapp/templates/
COPY checkpoint /Users/wentanbai/Desktop/code/myapp/
COPY model.ckpt.data-00000-of-00001 /Users/wentanbai/Desktop/code/myapp/
COPY model.ckpt.index /Users/wentanbai/Desktop/code/myapp/
COPY model.ckpt.meta /Users/wentanbai/Desktop/code/myapp/

# tell the port number the container should expose
EXPOSE 5000

# run the application
CMD ["python", "/Users/wentanbai/Desktop/code/myapp/app.py"]
