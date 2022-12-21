FROM python:3.9-slim

WORKDIR /root

RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD ../requirements.txt /root
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir LibRecommender
RUN pip install --no-cache-dir jupyterlab==3.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN jupyter server --generate-config --allow-root
# password generated based on https://jupyter-notebook.readthedocs.io/en/stable/config.html
RUN echo "c.ServerApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$1xV3ym3i6fh/Y9WrkfOfag\$pbATSK3YAwGw1GqdzGqhCw'" >> /root/.jupyter/jupyter_server_config.py
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_server_config.py
RUN echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_server_config.py

ADD ../examples /root/examples

EXPOSE 8888

CMD ["jupyter", "lab", "--allow-root", "--notebook-dir=/root", "--no-browser"]
