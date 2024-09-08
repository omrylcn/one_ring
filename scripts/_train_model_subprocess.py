# run_training_subprocess.py
import subprocess


# class TrainEnv():
#     def __init__(self) -> None:
#         pass


#     def start_training(self):
#         pass

#     def stop_training(self):
#         pass

#     def get_training_status(self):
#         pass

#     def start_tensorboard(self):
#         self.train_model(gamma=1.333, alpha=0.5, lr=0.001)

#     def stop_tensorboard(self):
#         pass

#     def start_mlflow(self):
#         pass

#     def stop_mlflow(self):
#         pass



import subprocess

def start_training_process():
    # Define the command to run your training script
    command = ["python", "train_model.py"]
    
    # Start the training script as a subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Optionally, wait for the process to complete and capture its output
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print("Training completed successfully.")
        print(stdout)
    else:
        print("Training process failed.")
        print(stderr)

# Example usage

start_training_process()