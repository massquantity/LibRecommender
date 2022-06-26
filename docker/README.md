# LibRecommender in Docker

Users can run [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) in a docker container and use the library without installing the package.

1. Pull image from Docker Hub
```shell
$ docker pull massquantity/librecommender:latest
```
2. Start a docker container by running following command:
```shell
$ docker run --rm -p 8889:8888 massquantity/librecommender:latest
```
This command exposes the container port 8888 to 8889 on your machine. Feel free to change 8889 to any port you want, 
but make sure it is available.

Or if you want to use your own data on your machine, try following command:
```shell
$ docker run --rm -p 8889:8888 -v $(pwd):/root/data:ro massquantity/librecommender:latest
```
The `-v` flag mounts the current directory to `/root/data` in the container, and the `ro` option means readonly. 
You can change `(pwd)` to the directory you want. For more information see [Use bind mounts](https://docs.docker.com/storage/bind-mounts/)

3. Open the JupyterLab in a browser with `http://localhost:8889`

4. Enter `LibRecommender` as the password.

5. The `examples` folder in the repository has been included in the container, so one can use the magic command in the notebook to run some example scripts:
```shell
cd examples
%run pure_ranking_example.py
```
