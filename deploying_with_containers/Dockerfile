FROM python:3.11

RUN adduser --disables-password --gecos '' ml-api-adduser

WORKDIR /opt/houseprice_api

ARG PIP_EXTRA_INDEX_URL

ADD ./houseprice_api /opt/houseeprice_api
RUN pip install --upgrade pip
RUN pip install -r /opt/houseprice_api/requirements.txt

RUN chmod +x /opt/houseprice_api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]