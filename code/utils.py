import os

# Helper functions
def make_sure_dir_exists(dir_to_check):
    if not os.path.exists(dir_to_check):
        os.makedirs(dir_to_check)
