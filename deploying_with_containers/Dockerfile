FROM python:3.11

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/houseprice_api

ARG PIP_EXTRA_INDEX_URL

ADD ./houseprice_api WORK_DIR # /opt/houseptrice_api
RUN pip install --upgrade pip
RUN pip install -r WORKDIR/requirements.txt # /opt/houseprice_api/requirements.txt

RUN chmod +x WORK_DIR/run.sh # /opt/houseprice_api/run.sh
RUN shown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]