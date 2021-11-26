FROM python:3.6-slim

LABEL maintainer="johannes.zerwas@tum.de"
LABEL version="1.0"
LABEL description="Basic python environment for cerberus-simulator"

ARG user=sim
ARG group=sim
ARG uid=1000
ARG gid=1000

# Add user
RUN addgroup --gid ${gid} ${group} && \
    adduser --home /home/${user} --uid ${uid} --ingroup ${group} --shell /bin/bash ${user}
RUN apt-get update && apt-get install -y wget git build-essential make

RUN pip install --upgrade pip setuptools wheel pyaml==18.11.0 networkx==2.2 numpy==1.17.0 peewee==3.8 pymysql==0.9.3  \
    sortedcontainers==2.1.0 pandas==0.24.2 tables \
    simplejson sphinx pytest gitpython

USER ${user}
CMD ["/bin/bash"]