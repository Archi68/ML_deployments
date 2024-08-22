FROM python:3.11

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/die_casting_api

ARG PIP_EXTRA_INDEX_URL

ADD ./die_casting_api /opt/die_casting_api
RUN pip install --upgrade pip
RUN pip install -r /opt/die_casting_api/requirements.txt

RUN chmod +x /opt/die_casting_api/run.sh

RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]