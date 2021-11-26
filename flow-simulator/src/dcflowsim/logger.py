import logging.config
import yaml
import os
import socket


def init_logging(logging_config=None, logging_output_dir=None, insert_hostname=False):
    """Init the logging file. Take the logging path as given in the init function.

    Args:
        logging_config: the path to the logging file.
        logging_output_dir: path to directory where logfiles should be stored. Logfilename in logging config is appended
            to this directory name
        insert_hostname
    """

    # If no logging path was given we take the standard one
    if logging_config is None:
        logging_config = os.path.join(
            os.path.dirname(__file__),
            "logging.yaml"
        )
    print(logging_config)
    if os.path.exists(logging_config):
        with open(logging_config, "rt") as logging_file:
            log_config = yaml.load(logging_file, Loader=yaml.FullLoader)

        # When we have a specific Logging output directory update the handler filenames to point to this dir
        for handler_name, handler_config in log_config["handlers"].items():
            if "filename" in handler_config.keys():
                if insert_hostname:
                    handler_config["filename"] = "{}_{}".format(socket.gethostname(), handler_config["filename"])
                if logging_output_dir is not None:
                    handler_config["filename"] = os.path.join(logging_output_dir, handler_config["filename"])
        logging.config.dictConfig(log_config)
    else:
        raise Exception("Error")
