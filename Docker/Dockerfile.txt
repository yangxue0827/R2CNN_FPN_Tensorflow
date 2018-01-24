FROM tensorflow/tensorflow:1.2.0-devel-gpu-py3

MAINTAINER TENSORFLOW DOCKER Maintainers 'yangxue0827@126.com'

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing -y && \ apt-get upgrade -y && \ apt-get install vim -y && \ apt-get install git -y && \ apt-get install python-opencv -y && \ apt-get install -y openssh-server

RUN mkdir /var/run/sshd RUN echo 'root:root' |chpasswd RUN sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN pip install opencv-python matplotlib && \ pip install keras

EXPOSE 22 EXPOSE 6006

CMD ["/usr/sbin/sshd", "-D"]