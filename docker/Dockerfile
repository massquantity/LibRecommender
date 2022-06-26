FROM python:3.9-slim

WORKDIR /root

RUN apt-get update && apt-get install -y gcc g++

ADD ../requirements.txt /root
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install LibRecommender
RUN pip install --no-cache-dir jupyterlab -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN jupyter notebook --generate-config --allow-root
# password generated based on https://jupyter-notebook.readthedocs.io/en/stable/config.html
RUN echo "c.NotebookApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$1xV3ym3i6fh/Y9WrkfOfag\$pbATSK3YAwGw1GqdzGqhCw'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py

ADD ../examples /root/examples

EXPOSE 8888

CMD ["jupyter", "lab", "--allow-root", "--notebook-dir=/root", "--no-browser"]
