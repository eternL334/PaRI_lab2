FROM python:3.11-slim

COPY requirements.txt /root/requirements.txt
RUN pip3 install -r /root/requirements.txt

COPY src/ /root/src/
RUN chown -R root:root /root/src

WORKDIR /root

CMD ["python3", "src/server.py"]