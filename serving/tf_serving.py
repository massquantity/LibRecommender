import os
import sys
import signal
import subprocess


if __name__ == "__main__":
    tf_server = subprocess.Popen(["sudo docker run -p 8501:8501 --mount type=bind,"
                                  "source=/home/massquantity/Workspace/LibRecommender/serving/models/FM,"
                                  "target=/models/FM -e MODEL_NAME=FM -t tensorflow/serving"],
                                 stdout=subprocess.DEVNULL,
                                 shell=True)
    print("tensorflow server started ...")

    flask_server = subprocess.Popen(["python deploy_feat_flask.py"], shell=True)
    print("flask server started ...")

    while True:
        print("Type 'exit' and press 'enter' to quit: ")
        in_str = input().strip().lower()
        if in_str == "q" or in_str == "exit":
            print("Shutting down all servers...")
        #    sys.exit(0)
            os.killpg(os.getpid(), signal.SIGTERM)
            break
        else:
            continue
